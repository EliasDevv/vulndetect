#!/usr/bin/env bash
# Run a queue of make targets unattended, one after another, each waiting for a
# free GPU (see gpu_wait_run.sh). Every target gets its own log, and a failing
# target does not abort the rest of the queue.
#
# Usage:
#   scripts/gpu_queue.sh [-g any|0|1|both] [-w LIST] [-i SEC] [-n CHECKS] TARGET [TARGET...]
#   scripts/gpu_queue.sh -f queue.txt          # one target per line, # = comment
#
#   -w  cards that must all be free before a target starts (see gpu_wait_run.sh);
#       "-g 0 -w 0,1" = run on card 0, but only once both cards are free.
#       Own processes never count as busy.
#
# Example:
#   tmux new -s queue -d 'cd ~/Bachelorarbeit/code && scripts/gpu_queue.sh \
#       dual_agent-smoke sft-smoke baseline-val rag-val dual_agent-val'
#
# Watch progress:  tail -f results/logs/queue_status.tsv
#                  tail -f results/logs/<target>_<timestamp>.log

set -uo pipefail # no -e: a failing target must not stop the queue

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WAIT=$HERE/gpu_wait_run.sh
LOGDIR=results/logs
STATUS=$LOGDIR/queue_status.tsv

GPUS=any
WATCH=
INTERVAL=60
STABLE=3
TARGETS=()

while [[ $# -gt 0 ]]; do
	case "$1" in
	-g | --gpu)
		GPUS=$2
		shift 2
		;;
	-w | --watch)
		WATCH=$2
		shift 2
		;;
	-i | --interval)
		INTERVAL=$2
		shift 2
		;;
	-n | --stable)
		STABLE=$2
		shift 2
		;;
	-f | --file)
		mapfile -t from_file < <(grep -vE '^\s*(#|$)' "$2")
		TARGETS+=("${from_file[@]}")
		shift 2
		;;
	-h | --help)
		sed -n '2,19p' "$0"
		exit 0
		;;
	*)
		TARGETS+=("$1")
		shift
		;;
	esac
done

[[ ${#TARGETS[@]} -gt 0 ]] || {
	echo "no targets given" >&2
	exit 2
}
[[ -x $WAIT ]] || {
	echo "missing $WAIT" >&2
	exit 1
}
[[ -f Makefile ]] || {
	echo "run me from code/ (the runners use relative results/ and data/ paths)" >&2
	exit 1
}

mkdir -p "$LOGDIR"
[[ -f $STATUS ]] || printf 'started\tfinished\ttarget\texit\tlog\n' >"$STATUS"

note() { echo "[$(date '+%F %T')] queue: $*"; }
trap 'note "interrupted -- stopping queue"; exit 130' INT TERM

note "${#TARGETS[@]} target(s) on GPU [$GPUS]: ${TARGETS[*]}"

for target in "${TARGETS[@]}"; do
	stamp=$(date '+%Y%m%d-%H%M%S')
	log=$LOGDIR/${target}_${stamp}.log
	start=$(date '+%F %T')
	note "starting '$target' -> $log"

	wait_opts=(-g "$GPUS" -i "$INTERVAL" -n "$STABLE")
	[[ -n $WATCH ]] && wait_opts+=(-w "$WATCH")
	"$WAIT" "${wait_opts[@]}" -- make "$target" >"$log" 2>&1
	rc=$?

	printf '%s\t%s\t%s\t%s\t%s\n' "$start" "$(date '+%F %T')" "$target" "$rc" "$log" >>"$STATUS"
	if [[ $rc -eq 0 ]]; then
		note "'$target' OK"
	else
		note "'$target' FAILED (exit $rc) -- continuing with the rest, see $log"
	fi
done

note "queue done. Summary:"
column -t -s $'\t' "$STATUS"

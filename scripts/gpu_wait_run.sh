#!/usr/bin/env bash
# Wait for a free GPU, then run a command on it.
#
# Polls nvidia-smi until the requested GPU(s) have been idle for several
# consecutive checks, then exports CUDA_VISIBLE_DEVICES and execs the command.
# Processes of the calling user are ignored. Run it inside tmux.
#
# Usage:
#   scripts/gpu_wait_run.sh [-g 0|1|any|both] [-w LIST] [-m MiB] [-i SEC] [-n CHECKS] -- CMD...
#
#   -g  which GPU to run on: an index, "any" (first one to free up, default)
#       or "both" (wait until all candidates are free at the same time)
#   -w  which GPUs to watch: comma-separated list that must all be free before
#       launching (default: the -g card alone). Use "-w 0,1" when a neighbour's
#       job spans both cards and you want to wait it out completely.
#   -m  a GPU counts as free below this much foreign VRAM (default 1024 MiB)
#   -i  seconds between checks (default 60; a small random jitter is added)
#   -n  consecutive free checks required before launching (default 3)
#   -T  give up after this many seconds of waiting (default: wait forever);
#       on timeout the script exits 75 (EX_TEMPFAIL) without running CMD
#
# Examples:
#   scripts/gpu_wait_run.sh -- make baseline-val
#   scripts/gpu_wait_run.sh -g 1 -- make rag-val
#   scripts/gpu_wait_run.sh -g 0 -w 0,1 -- make baseline-val   # wait out both
#   scripts/gpu_wait_run.sh -g both -- make sft-train
#   scripts/gpu_wait_run.sh -- bash -c 'make baseline && make rag'   # queue

set -euo pipefail

GPUS=any
WATCH=
FREE_MIB=1024
INTERVAL=60
STABLE=3
MAX_WAIT=0

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
	-m | --free-mib)
		FREE_MIB=$2
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
	-T | --max-wait)
		MAX_WAIT=$2
		shift 2
		;;
	-h | --help)
		sed -n '2,27p' "$0"
		exit 0
		;;
	--)
		shift
		break
		;;
	*)
		echo "unknown option: $1 (use -- before the command)" >&2
		exit 2
		;;
	esac
done

[[ $# -gt 0 ]] || {
	echo "no command given (use -- CMD...)" >&2
	exit 2
}
command -v nvidia-smi >/dev/null || {
	echo "nvidia-smi not found" >&2
	exit 1
}

ALL_IDX=$(nvidia-smi --query-gpu=index --format=csv,noheader)
case "$GPUS" in
any | both) CANDIDATES=$ALL_IDX ;;
*) CANDIDATES=${GPUS//,/ } ;;
esac

WATCHED=${WATCH//,/ }
[[ -n $WATCHED ]] || WATCHED=$CANDIDATES
MONITORED=$(printf '%s\n' $CANDIDATES $WATCHED | sort -un)

MY_UID=$(id -u)

log() { echo "[$(date '+%F %T')] $*"; }

gpu_used_mib() { nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits; }

# Prints "<foreign procs> <own MiB>" for GPU $1; the caller subtracts own MiB from memory.used.
gpu_own() {
	local uuid guuid pid mem owner nprocs=0 mib=0
	uuid=$(nvidia-smi --id="$1" --query-gpu=uuid --format=csv,noheader)
	while IFS=, read -r guuid pid mem; do
		guuid=${guuid// /}
		pid=${pid// /}
		mem=${mem// /}
		[[ $guuid == "$uuid" && -n $pid ]] || continue
		owner=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
		[[ -z $owner ]] && continue
		if [[ $owner == "$MY_UID" ]]; then
			[[ $mem =~ ^[0-9]+$ ]] && mib=$((mib + mem))
		else
			nprocs=$((nprocs + 1))
		fi
	done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
		--format=csv,noheader,nounits 2>/dev/null)
	echo "$nprocs $mib"
}

declare -A streak
for i in $MONITORED; do streak[$i]=0; done

log "run on GPU(s) [$GPUS], watching [$(echo $WATCHED | tr '\n' ' ')] of [$(echo "$ALL_IDX" | tr '\n' ' ')]"
log "free = 0 foreign procs and <${FREE_MIB} MiB foreign VRAM, ${STABLE}x every ${INTERVAL}s (own processes ignored)"
log "command: $*"

last_status=""
started=$SECONDS
while true; do
	if [[ $MAX_WAIT -gt 0 && $((SECONDS - started)) -ge $MAX_WAIT ]]; then
		log "gave up after ${MAX_WAIT}s -- no free GPU, command NOT run"
		exit 75
	fi
	status=""
	for i in $MONITORED; do
		used=$(gpu_used_mib "$i")
		read -r procs ownmib < <(gpu_own "$i")
		fmib=$((used - ownmib))
		((fmib < 0)) && fmib=0
		if [[ $procs -eq 0 && $fmib -lt $FREE_MIB ]]; then
			streak[$i]=$((streak[$i] + 1))
		else
			streak[$i]=0
		fi
		status+="gpu$i: ${used}MiB used/${fmib}MiB foreign/${procs}proc(free ${streak[$i]}/${STABLE})  "
	done

	# Every watched card must be free before anything launches.
	gate=1
	for i in $WATCHED; do [[ ${streak[$i]} -ge $STABLE ]] || gate=0; done

	target=""
	if [[ $gate -eq 0 ]]; then
		:
	elif [[ $GPUS == both ]]; then
		all_ready=1
		for i in $CANDIDATES; do [[ ${streak[$i]} -ge $STABLE ]] || all_ready=0; done
		[[ $all_ready -eq 1 ]] && target=$(echo "$CANDIDATES" | tr '\n' ',' | sed 's/ /,/g; s/,$//')
	else
		for i in $CANDIDATES; do
			if [[ ${streak[$i]} -ge $STABLE ]]; then
				target=$i
				break
			fi
		done
	fi

	if [[ -n $target ]]; then
		log "GPU $target free -> launching"
		export CUDA_VISIBLE_DEVICES="$target"
		exec "$@"
	fi

	# Only log on change.
	[[ $status != "$last_status" ]] && log "$status"
	last_status=$status
	sleep $((INTERVAL + RANDOM % 15))
done

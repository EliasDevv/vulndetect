# Comparison — split: val

Parser: d09-v2. Primary unparseable policy: drop; the alt row is the other policy's macro. Values are mean±std over seeds.

## All (Avg) overview

| approach | seeds | accuracy | precision | recall | f1 | f1 (alt) | parse-fail |
|---|---|---|---|---|---|---|---|
| baseline_val | 42 | 0.580 | 0.632 | 0.410 | 0.492 | 0.484 | 0.012 |
| rag_val | 42 | 0.597 | 0.649 | 0.456 | 0.527 | 0.514 | 0.024 |
| dual_agent_val | 42 | 0.527 | 0.531 | 0.430 | 0.473 | 0.473 | 0.000 |
| sft_val | 42 | 0.681 | 0.655 | 0.784 | 0.712 | 0.712 | 0.000 |
| baseline_val (paper) | — | 0.650 | 0.680 | 0.660 | 0.670 | — | — |
| rag_val (paper) | — | 0.860 | 0.880 | 0.840 | 0.850 | — | — |
| sft_val (paper) | — | 0.810 | 0.850 | 0.760 | 0.800 | — | — |
| dual_agent_val (paper) | — | 0.780 | 0.800 | 0.760 | 0.770 | — | — |

## baseline_val — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 278 | 0.607 | 0.697 | 0.384 | 0.495 |
| CWE-20 | 88 | 0.655 | 0.741 | 0.465 | 0.571 |
| CWE-200 | 46 | 0.356 | 0.333 | 0.261 | 0.293 |
| CWE-264 | 46 | 0.733 | 0.789 | 0.652 | 0.714 |
| CWE-399 | 42 | 0.548 | 0.600 | 0.286 | 0.387 |
| All (macro) | 500 | 0.580 | 0.632 | 0.410 | 0.492 |
| All (micro) | 500 | 0.599 | 0.667 | 0.403 | 0.503 |

## rag_val — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 278 | 0.576 | 0.638 | 0.331 | 0.436 |
| CWE-20 | 88 | 0.602 | 0.629 | 0.500 | 0.557 |
| CWE-200 | 46 | 0.500 | 0.500 | 0.478 | 0.489 |
| CWE-264 | 46 | 0.733 | 0.778 | 0.636 | 0.700 |
| CWE-399 | 42 | 0.575 | 0.700 | 0.333 | 0.452 |
| All (macro) | 500 | 0.597 | 0.649 | 0.456 | 0.527 |
| All (micro) | 500 | 0.588 | 0.636 | 0.403 | 0.494 |

## dual_agent_val — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 278 | 0.529 | 0.539 | 0.396 | 0.456 |
| CWE-20 | 88 | 0.466 | 0.457 | 0.364 | 0.405 |
| CWE-200 | 46 | 0.565 | 0.565 | 0.565 | 0.565 |
| CWE-264 | 46 | 0.478 | 0.471 | 0.348 | 0.400 |
| CWE-399 | 42 | 0.595 | 0.625 | 0.476 | 0.541 |
| All (macro) | 500 | 0.527 | 0.531 | 0.430 | 0.473 |
| All (micro) | 500 | 0.522 | 0.528 | 0.408 | 0.460 |

## sft_val — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 278 | 0.608 | 0.594 | 0.683 | 0.635 |
| CWE-20 | 88 | 0.693 | 0.667 | 0.773 | 0.716 |
| CWE-200 | 46 | 0.609 | 0.581 | 0.783 | 0.667 |
| CWE-264 | 46 | 0.804 | 0.792 | 0.826 | 0.809 |
| CWE-399 | 42 | 0.690 | 0.643 | 0.857 | 0.735 |
| All (macro) | 500 | 0.681 | 0.655 | 0.784 | 0.712 |
| All (micro) | 500 | 0.648 | 0.626 | 0.736 | 0.676 |

## Paired t-tests (per-CWE F1 means, paper's test; df = #CWEs - 1)

| a | b | mean diff (a-b) | t | p | 95% CI |
|---|---|---|---|---|---|
| baseline_val | dual_agent_val | +0.019 | 0.177 | 0.8682 | [-0.275, +0.313] |
| baseline_val | rag_val | -0.034 | -0.764 | 0.4875 | [-0.160, +0.091] |
| baseline_val | sft_val | -0.220 | -3.776 | 0.0195 | [-0.382, -0.058] |
| dual_agent_val | rag_val | -0.053 | -0.707 | 0.5185 | [-0.262, +0.156] |
| dual_agent_val | sft_val | -0.239 | -4.418 | 0.0115 | [-0.389, -0.089] |
| rag_val | sft_val | -0.186 | -6.477 | 0.0029 | [-0.265, -0.106] |

## McNemar (instance-level, per seed-matched pair; re-parsed with the parser above, unparseable pairs excluded under drop)

| a | b | seed | n | excluded | b(a✓b✗) | c(a✗b✓) | p | dir | p (sensitivity) |
|---|---|---|---|---|---|---|---|---|---|
| baseline_val | dual_agent_val | 42 | 494 | 6 | 149 | 111 | 0.0216 | a>b | 0.0358 (wrong) |
| baseline_val | rag_val | 42 | 482 | 18 | 76 | 72 | 0.8053 | a>b | 0.5310 (wrong) |
| baseline_val | sft_val | 42 | 494 | 6 | 96 | 121 | 0.1030 | b>a | 0.0685 (wrong) |
| dual_agent_val | rag_val | 42 | 488 | 12 | 100 | 128 | 0.0735 | b>a | 0.0990 (wrong) |
| dual_agent_val | sft_val | 42 | 500 | 0 | 106 | 169 | 0.0002 | b>a | 0.0002 (wrong) |
| rag_val | sft_val | 42 | 488 | 12 | 97 | 128 | 0.0453 | b>a | 0.0177 (wrong) |

## Seed rule (Section 6.6): p_max over seeds, direction in every seed, alpha = 0.05; p_holm within named families

| a | b | seeds | direction | p_max | p_holm | significant | p_max (sensitivity) | significant (sensitivity) |
|---|---|---|---|---|---|---|---|---|
| baseline_val | dual_agent_val | 42 | a>b | 0.0216 | — | True | 0.0358 (wrong) | True |
| baseline_val | rag_val | 42 | a>b | 0.8053 | — | False | 0.5310 (wrong) | False |
| baseline_val | sft_val | 42 | b>a | 0.1030 | — | False | 0.0685 (wrong) | False |
| dual_agent_val | rag_val | 42 | b>a | 0.0735 | — | False | 0.0990 (wrong) | False |
| dual_agent_val | sft_val | 42 | b>a | 0.0002 | — | True | 0.0002 (wrong) | True |
| rag_val | sft_val | 42 | b>a | 0.0453 | — | True | 0.0177 (wrong) | True |

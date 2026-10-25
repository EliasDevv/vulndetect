# Comparison — split: test

Parser: d09-v2. Primary unparseable policy: drop; the alt row is the other policy's macro. Values are mean±std over seeds.

## All (Avg) overview

| approach | seeds | accuracy | precision | recall | f1 | f1 (alt) | parse-fail |
|---|---|---|---|---|---|---|---|
| baseline | 42,43,44 | 0.576±0.020 | 0.617±0.025 | 0.455±0.026 | 0.517±0.026 | 0.512±0.025 | 0.008±0.002 |
| rag | 42,43,44 | 0.576±0.019 | 0.604±0.025 | 0.491±0.021 | 0.535±0.024 | 0.525±0.026 | 0.018±0.004 |
| sft | 42,43,44 | 0.657±0.015 | 0.636±0.022 | 0.761±0.038 | 0.688±0.010 | 0.688±0.010 | 0.000±0.000 |
| dual_agent | 42,43,44 | 0.521±0.011 | 0.526±0.014 | 0.421±0.010 | 0.467±0.012 | 0.467±0.012 | 0.000±0.000 |
| gan/round0000 | 42,43,44 | 0.576±0.020 | 0.617±0.025 | 0.455±0.026 | 0.517±0.026 | 0.512±0.025 | 0.008±0.002 |
| gan/round0010 | 42 | 0.602 | 0.638 | 0.488 | 0.546 | 0.544 | 0.005 |
| gan/round0050 | 42 | 0.586 | 0.615 | 0.485 | 0.537 | 0.533 | 0.008 |
| gan/round0200 | 42 | 0.565 | 0.588 | 0.470 | 0.510 | 0.503 | 0.013 |
| gan/final | 42,43,44 | 0.595±0.008 | 0.632±0.014 | 0.512±0.021 | 0.555±0.012 | 0.553±0.011 | 0.005±0.001 |
| triple_agent/round0000 | 42,43,44 | 0.521±0.011 | 0.526±0.014 | 0.421±0.010 | 0.467±0.012 | 0.467±0.012 | 0.000±0.000 |
| triple_agent/final | 42,43,44 | 0.549±0.014 | 0.563±0.015 | 0.451±0.024 | 0.499±0.021 | 0.498±0.021 | 0.001±0.001 |

## baseline — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 340 | 0.589±0.046 | 0.627±0.060 | 0.448±0.055 | 0.522±0.058 |
| CWE-20 | 132 | 0.646±0.024 | 0.780±0.009 | 0.400±0.067 | 0.526±0.061 |
| CWE-200 | 208 | 0.561±0.029 | 0.567±0.032 | 0.531±0.034 | 0.548±0.031 |
| CWE-264 | 100 | 0.503±0.043 | 0.499±0.046 | 0.456±0.062 | 0.476±0.055 |
| CWE-399 | 206 | 0.579±0.025 | 0.612±0.022 | 0.443±0.050 | 0.513±0.041 |
| All (macro) | 986 | 0.576±0.020 | 0.617±0.025 | 0.455±0.026 | 0.517±0.026 |
| All (micro) | 986 | 0.580±0.023 | 0.607±0.031 | 0.459±0.028 | 0.523±0.029 |

## rag — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 340 | 0.598±0.028 | 0.623±0.035 | 0.511±0.022 | 0.561±0.027 |
| CWE-20 | 132 | 0.609±0.010 | 0.679±0.031 | 0.401±0.047 | 0.503±0.036 |
| CWE-200 | 208 | 0.528±0.024 | 0.530±0.035 | 0.501±0.068 | 0.514±0.049 |
| CWE-264 | 100 | 0.513±0.046 | 0.504±0.044 | 0.544±0.077 | 0.523±0.059 |
| CWE-399 | 206 | 0.633±0.025 | 0.686±0.042 | 0.497±0.052 | 0.575±0.038 |
| All (macro) | 986 | 0.576±0.019 | 0.604±0.025 | 0.491±0.021 | 0.535±0.024 |
| All (micro) | 986 | 0.583±0.018 | 0.602±0.026 | 0.495±0.012 | 0.543±0.018 |

## sft — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 340 | 0.598±0.007 | 0.578±0.012 | 0.735±0.046 | 0.646±0.011 |
| CWE-20 | 132 | 0.692±0.016 | 0.719±0.042 | 0.636±0.052 | 0.673±0.019 |
| CWE-200 | 208 | 0.646±0.019 | 0.612±0.020 | 0.804±0.078 | 0.693±0.026 |
| CWE-264 | 100 | 0.680±0.046 | 0.641±0.039 | 0.820±0.035 | 0.720±0.037 |
| CWE-399 | 206 | 0.668±0.047 | 0.632±0.039 | 0.809±0.059 | 0.709±0.039 |
| All (macro) | 986 | 0.657±0.015 | 0.636±0.022 | 0.761±0.038 | 0.688±0.010 |
| All (micro) | 986 | 0.644±0.014 | 0.617±0.019 | 0.761±0.046 | 0.681±0.013 |

## dual_agent — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 340 | 0.514±0.014 | 0.517±0.017 | 0.414±0.021 | 0.460±0.019 |
| CWE-20 | 132 | 0.513±0.012 | 0.516±0.014 | 0.409±0.015 | 0.456±0.015 |
| CWE-200 | 208 | 0.516±0.010 | 0.520±0.013 | 0.410±0.015 | 0.459±0.014 |
| CWE-264 | 100 | 0.563±0.038 | 0.572±0.044 | 0.507±0.031 | 0.537±0.035 |
| CWE-399 | 206 | 0.502±0.012 | 0.502±0.018 | 0.366±0.030 | 0.423±0.022 |
| All (macro) | 986 | 0.521±0.011 | 0.526±0.014 | 0.421±0.010 | 0.467±0.012 |
| All (micro) | 986 | 0.517±0.010 | 0.521±0.013 | 0.412±0.009 | 0.460±0.011 |

## gan/round0000 — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 340 | 0.589±0.046 | 0.627±0.060 | 0.448±0.055 | 0.522±0.058 |
| CWE-20 | 132 | 0.646±0.024 | 0.780±0.009 | 0.400±0.067 | 0.526±0.061 |
| CWE-200 | 208 | 0.561±0.029 | 0.567±0.032 | 0.531±0.034 | 0.548±0.031 |
| CWE-264 | 100 | 0.503±0.043 | 0.499±0.046 | 0.456±0.062 | 0.476±0.055 |
| CWE-399 | 206 | 0.579±0.025 | 0.612±0.022 | 0.443±0.050 | 0.513±0.041 |
| All (macro) | 986 | 0.576±0.020 | 0.617±0.025 | 0.455±0.026 | 0.517±0.026 |
| All (micro) | 986 | 0.580±0.023 | 0.607±0.031 | 0.459±0.028 | 0.523±0.029 |

## gan/round0010 — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 340 | 0.638 | 0.672 | 0.536 | 0.596 |
| CWE-20 | 132 | 0.595 | 0.697 | 0.348 | 0.465 |
| CWE-200 | 208 | 0.620 | 0.621 | 0.615 | 0.618 |
| CWE-264 | 100 | 0.550 | 0.558 | 0.480 | 0.516 |
| CWE-399 | 206 | 0.605 | 0.644 | 0.461 | 0.537 |
| All (macro) | 986 | 0.602 | 0.638 | 0.488 | 0.546 |
| All (micro) | 986 | 0.613 | 0.642 | 0.506 | 0.566 |

## gan/round0050 — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 340 | 0.602 | 0.627 | 0.500 | 0.556 |
| CWE-20 | 132 | 0.664 | 0.775 | 0.470 | 0.585 |
| CWE-200 | 208 | 0.663 | 0.657 | 0.683 | 0.670 |
| CWE-264 | 100 | 0.465 | 0.462 | 0.360 | 0.404 |
| CWE-399 | 206 | 0.537 | 0.553 | 0.412 | 0.472 |
| All (macro) | 986 | 0.586 | 0.615 | 0.485 | 0.537 |
| All (micro) | 986 | 0.596 | 0.620 | 0.502 | 0.555 |

## gan/round0200 — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 340 | 0.588 | 0.607 | 0.521 | 0.561 |
| CWE-20 | 132 | 0.531 | 0.586 | 0.258 | 0.358 |
| CWE-200 | 208 | 0.637 | 0.643 | 0.618 | 0.630 |
| CWE-264 | 100 | 0.510 | 0.509 | 0.580 | 0.542 |
| CWE-399 | 206 | 0.559 | 0.594 | 0.373 | 0.458 |
| All (macro) | 986 | 0.565 | 0.588 | 0.470 | 0.510 |
| All (micro) | 986 | 0.577 | 0.598 | 0.481 | 0.533 |

## gan/final — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 340 | 0.616±0.008 | 0.639±0.012 | 0.537±0.063 | 0.582±0.034 |
| CWE-20 | 132 | 0.637±0.028 | 0.772±0.043 | 0.394±0.055 | 0.520±0.054 |
| CWE-200 | 208 | 0.601±0.040 | 0.592±0.032 | 0.655±0.065 | 0.621±0.042 |
| CWE-264 | 100 | 0.523±0.023 | 0.524±0.023 | 0.500±0.040 | 0.512±0.031 |
| CWE-399 | 206 | 0.599±0.021 | 0.634±0.040 | 0.474±0.010 | 0.542±0.009 |
| All (macro) | 986 | 0.595±0.008 | 0.632±0.014 | 0.512±0.021 | 0.555±0.012 |
| All (micro) | 986 | 0.603±0.007 | 0.623±0.015 | 0.525±0.018 | 0.570±0.005 |

## triple_agent/round0000 — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 340 | 0.514±0.014 | 0.517±0.017 | 0.414±0.021 | 0.460±0.019 |
| CWE-20 | 132 | 0.513±0.012 | 0.516±0.014 | 0.409±0.015 | 0.456±0.015 |
| CWE-200 | 208 | 0.516±0.010 | 0.520±0.013 | 0.410±0.015 | 0.459±0.014 |
| CWE-264 | 100 | 0.563±0.038 | 0.572±0.044 | 0.507±0.031 | 0.537±0.035 |
| CWE-399 | 206 | 0.502±0.012 | 0.502±0.018 | 0.366±0.030 | 0.423±0.022 |
| All (macro) | 986 | 0.521±0.011 | 0.526±0.014 | 0.421±0.010 | 0.467±0.012 |
| All (micro) | 986 | 0.517±0.010 | 0.521±0.013 | 0.412±0.009 | 0.460±0.011 |

## triple_agent/final — per CWE

| CWE | n | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| CWE-119 | 340 | 0.562±0.049 | 0.570±0.055 | 0.502±0.069 | 0.534±0.063 |
| CWE-20 | 132 | 0.583±0.015 | 0.621±0.023 | 0.429±0.053 | 0.506±0.037 |
| CWE-200 | 208 | 0.506±0.014 | 0.506±0.017 | 0.441±0.026 | 0.471±0.020 |
| CWE-264 | 100 | 0.550±0.026 | 0.563±0.033 | 0.453±0.076 | 0.500±0.049 |
| CWE-399 | 206 | 0.543±0.041 | 0.557±0.058 | 0.429±0.018 | 0.484±0.032 |
| All (macro) | 986 | 0.549±0.014 | 0.563±0.015 | 0.451±0.024 | 0.499±0.021 |
| All (micro) | 986 | 0.548±0.019 | 0.558±0.023 | 0.459±0.026 | 0.504±0.025 |

## Paired t-tests (per-CWE F1 means, paper's test; df = #CWEs - 1)

| a | b | mean diff (a-b) | t | p | 95% CI |
|---|---|---|---|---|---|
| baseline | dual_agent | +0.050 | 1.770 | 0.1514 | [-0.029, +0.129] |
| baseline | gan/final | -0.038 | -2.803 | 0.0486 | [-0.076, -0.000] |
| baseline | gan/round0000 | +0.000 | nan | nan | [+0.000, +0.000] |
| baseline | gan/round0010 | -0.029 | -1.191 | 0.2996 | [-0.097, +0.039] |
| baseline | gan/round0050 | -0.020 | -0.582 | 0.5917 | [-0.117, +0.076] |
| baseline | gan/round0200 | +0.008 | 0.162 | 0.8791 | [-0.122, +0.137] |
| baseline | rag | -0.018 | -0.933 | 0.4038 | [-0.072, +0.036] |
| baseline | sft | -0.171 | -7.923 | 0.0014 | [-0.231, -0.111] |
| baseline | triple_agent/final | +0.018 | 1.037 | 0.3583 | [-0.031, +0.067] |
| baseline | triple_agent/round0000 | +0.050 | 1.770 | 0.1514 | [-0.029, +0.129] |
| dual_agent | gan/final | -0.088 | -2.717 | 0.0531 | [-0.179, +0.002] |
| dual_agent | gan/round0000 | -0.050 | -1.770 | 0.1514 | [-0.129, +0.029] |
| dual_agent | gan/round0010 | -0.080 | -2.202 | 0.0924 | [-0.180, +0.021] |
| dual_agent | gan/round0050 | -0.071 | -1.231 | 0.2856 | [-0.230, +0.089] |
| dual_agent | gan/round0200 | -0.043 | -0.939 | 0.4007 | [-0.169, +0.084] |
| dual_agent | rag | -0.068 | -2.450 | 0.0704 | [-0.146, +0.009] |
| dual_agent | sft | -0.221 | -11.708 | 0.0003 | [-0.274, -0.169] |
| dual_agent | triple_agent/final | -0.032 | -1.580 | 0.1892 | [-0.088, +0.024] |
| dual_agent | triple_agent/round0000 | +0.000 | nan | nan | [+0.000, +0.000] |
| gan/final | gan/round0000 | +0.038 | 2.803 | 0.0486 | [+0.000, +0.076] |
| gan/final | gan/round0010 | +0.009 | 0.736 | 0.5027 | [-0.025, +0.043] |
| gan/final | gan/round0050 | +0.018 | 0.542 | 0.6164 | [-0.074, +0.110] |
| gan/final | gan/round0200 | +0.046 | 1.308 | 0.2610 | [-0.051, +0.143] |
| gan/final | rag | +0.020 | 0.835 | 0.4508 | [-0.047, +0.087] |
| gan/final | sft | -0.133 | -4.760 | 0.0089 | [-0.210, -0.055] |
| gan/final | triple_agent/final | +0.057 | 2.240 | 0.0886 | [-0.014, +0.127] |
| gan/final | triple_agent/round0000 | +0.088 | 2.717 | 0.0531 | [-0.002, +0.179] |
| gan/round0000 | gan/round0010 | -0.029 | -1.191 | 0.2996 | [-0.097, +0.039] |
| gan/round0000 | gan/round0050 | -0.020 | -0.582 | 0.5917 | [-0.117, +0.076] |
| gan/round0000 | gan/round0200 | +0.008 | 0.162 | 0.8791 | [-0.122, +0.137] |
| gan/round0000 | rag | -0.018 | -0.933 | 0.4038 | [-0.072, +0.036] |
| gan/round0000 | sft | -0.171 | -7.923 | 0.0014 | [-0.231, -0.111] |
| gan/round0000 | triple_agent/final | +0.018 | 1.037 | 0.3583 | [-0.031, +0.067] |
| gan/round0000 | triple_agent/round0000 | +0.050 | 1.770 | 0.1514 | [-0.029, +0.129] |
| gan/round0010 | gan/round0050 | +0.009 | 0.215 | 0.8406 | [-0.107, +0.125] |
| gan/round0010 | gan/round0200 | +0.037 | 1.444 | 0.2221 | [-0.034, +0.108] |
| gan/round0010 | rag | +0.011 | 0.410 | 0.7026 | [-0.064, +0.086] |
| gan/round0010 | sft | -0.142 | -4.276 | 0.0129 | [-0.234, -0.050] |
| gan/round0010 | triple_agent/final | +0.048 | 1.535 | 0.1996 | [-0.038, +0.134] |
| gan/round0010 | triple_agent/round0000 | +0.080 | 2.202 | 0.0924 | [-0.021, +0.180] |
| gan/round0050 | gan/round0200 | +0.028 | 0.476 | 0.6590 | [-0.135, +0.190] |
| gan/round0050 | rag | +0.002 | 0.039 | 0.9706 | [-0.145, +0.149] |
| gan/round0050 | sft | -0.151 | -2.796 | 0.0490 | [-0.301, -0.001] |
| gan/round0050 | triple_agent/final | +0.039 | 0.786 | 0.4756 | [-0.098, +0.175] |
| gan/round0050 | triple_agent/round0000 | +0.071 | 1.231 | 0.2856 | [-0.089, +0.230] |
| gan/round0200 | rag | -0.026 | -0.540 | 0.6177 | [-0.158, +0.107] |
| gan/round0200 | sft | -0.179 | -3.731 | 0.0203 | [-0.312, -0.046] |
| gan/round0200 | triple_agent/final | +0.011 | 0.215 | 0.8401 | [-0.128, +0.150] |
| gan/round0200 | triple_agent/round0000 | +0.043 | 0.939 | 0.4007 | [-0.084, +0.169] |
| rag | sft | -0.153 | -7.707 | 0.0015 | [-0.208, -0.098] |
| rag | triple_agent/final | +0.037 | 2.348 | 0.0787 | [-0.007, +0.080] |
| rag | triple_agent/round0000 | +0.068 | 2.450 | 0.0704 | [-0.009, +0.146] |
| sft | triple_agent/final | +0.189 | 8.603 | 0.0010 | [+0.128, +0.251] |
| sft | triple_agent/round0000 | +0.221 | 11.708 | 0.0003 | [+0.169, +0.274] |
| triple_agent/final | triple_agent/round0000 | +0.032 | 1.580 | 0.1892 | [-0.024, +0.088] |

## McNemar (instance-level, per seed-matched pair; re-parsed with the parser above, unparseable pairs excluded under drop)

| a | b | seed | n | excluded | b(a✓b✗) | c(a✗b✓) | p | dir | p (sensitivity) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | dual_agent | 42 | 977 | 9 | 319 | 233 | 0.0003 | a>b | 0.0007 (wrong) |
| baseline | dual_agent | 43 | 980 | 6 | 299 | 267 | 0.1925 | a>b | 0.2236 (wrong) |
| baseline | dual_agent | 44 | 976 | 10 | 307 | 238 | 0.0035 | a>b | 0.0082 (wrong) |
| baseline | gan/final | 42 | 973 | 13 | 160 | 156 | 0.8660 | a>b | 1.0000 (wrong) |
| baseline | gan/final | 43 | 975 | 11 | 138 | 187 | 0.0077 | b>a | 0.0069 (wrong) |
| baseline | gan/final | 44 | 970 | 16 | 153 | 175 | 0.2462 | b>a | 0.2519 (wrong) |
| baseline | gan/round0000 | 42 | 977 | 9 | 0 | 0 | 1.0000 | tie | 1.0000 (wrong) |
| baseline | gan/round0000 | 43 | 980 | 6 | 0 | 0 | 1.0000 | tie | 1.0000 (wrong) |
| baseline | gan/round0000 | 44 | 976 | 10 | 0 | 0 | 1.0000 | tie | 1.0000 (wrong) |
| baseline | gan/round0010 | 42 | 972 | 14 | 165 | 172 | 0.7438 | b>a | 0.6286 (wrong) |
| baseline | gan/round0050 | 42 | 969 | 17 | 176 | 165 | 0.5882 | a>b | 0.7091 (wrong) |
| baseline | gan/round0200 | 42 | 965 | 21 | 175 | 147 | 0.1323 | a>b | 0.1135 (wrong) |
| baseline | rag | 42 | 964 | 22 | 175 | 173 | 0.9573 | a>b | 0.7928 (wrong) |
| baseline | rag | 43 | 964 | 22 | 165 | 170 | 0.8270 | b>a | 1.0000 (wrong) |
| baseline | rag | 44 | 955 | 31 | 162 | 169 | 0.7416 | b>a | 1.0000 (wrong) |
| baseline | sft | 42 | 977 | 9 | 205 | 252 | 0.0313 | b>a | 0.0122 (wrong) |
| baseline | sft | 43 | 980 | 6 | 203 | 267 | 0.0036 | b>a | 0.0015 (wrong) |
| baseline | sft | 44 | 976 | 10 | 189 | 259 | 0.0011 | b>a | 0.0004 (wrong) |
| baseline | triple_agent/final | 42 | 975 | 11 | 257 | 179 | 0.0002 | a>b | 0.0006 (wrong) |
| baseline | triple_agent/final | 43 | 979 | 7 | 195 | 198 | 0.9197 | b>a | 0.8409 (wrong) |
| baseline | triple_agent/final | 44 | 976 | 10 | 239 | 219 | 0.3747 | a>b | 0.5153 (wrong) |
| baseline | triple_agent/round0000 | 42 | 977 | 9 | 319 | 233 | 0.0003 | a>b | 0.0007 (wrong) |
| baseline | triple_agent/round0000 | 43 | 980 | 6 | 299 | 267 | 0.1925 | a>b | 0.2236 (wrong) |
| baseline | triple_agent/round0000 | 44 | 976 | 10 | 307 | 238 | 0.0035 | a>b | 0.0082 (wrong) |
| dual_agent | gan/final | 42 | 982 | 4 | 216 | 300 | 0.0003 | b>a | 0.0004 (wrong) |
| dual_agent | gan/final | 43 | 981 | 5 | 196 | 278 | 0.0002 | b>a | 0.0003 (wrong) |
| dual_agent | gan/final | 44 | 980 | 6 | 196 | 284 | 0.0001 | b>a | 0.0001 (wrong) |
| dual_agent | gan/round0000 | 42 | 977 | 9 | 233 | 319 | 0.0003 | b>a | 0.0007 (wrong) |
| dual_agent | gan/round0000 | 43 | 980 | 6 | 267 | 299 | 0.1925 | b>a | 0.2236 (wrong) |
| dual_agent | gan/round0000 | 44 | 976 | 10 | 238 | 307 | 0.0035 | b>a | 0.0082 (wrong) |
| dual_agent | gan/round0010 | 42 | 981 | 5 | 194 | 287 | 0.0000 | b>a | 0.0000 (wrong) |
| dual_agent | gan/round0050 | 42 | 978 | 8 | 203 | 281 | 0.0005 | b>a | 0.0011 (wrong) |
| dual_agent | gan/round0200 | 42 | 973 | 13 | 201 | 257 | 0.0101 | b>a | 0.0200 (wrong) |
| dual_agent | rag | 42 | 972 | 14 | 191 | 275 | 0.0001 | b>a | 0.0007 (wrong) |
| dual_agent | rag | 43 | 969 | 17 | 216 | 254 | 0.0878 | b>a | 0.1846 (wrong) |
| dual_agent | rag | 44 | 965 | 21 | 190 | 264 | 0.0006 | b>a | 0.0047 (wrong) |
| dual_agent | sft | 42 | 986 | 0 | 192 | 328 | 0.0000 | b>a | 0.0000 (wrong) |
| dual_agent | sft | 43 | 986 | 0 | 220 | 320 | 0.0000 | b>a | 0.0000 (wrong) |
| dual_agent | sft | 44 | 986 | 0 | 189 | 329 | 0.0000 | b>a | 0.0000 (wrong) |
| dual_agent | triple_agent/final | 42 | 984 | 2 | 227 | 237 | 0.6761 | b>a | 0.7458 (wrong) |
| dual_agent | triple_agent/final | 43 | 985 | 1 | 223 | 259 | 0.1108 | b>a | 0.1218 (wrong) |
| dual_agent | triple_agent/final | 44 | 986 | 0 | 202 | 250 | 0.0269 | b>a | 0.0269 (wrong) |
| dual_agent | triple_agent/round0000 | 42 | 986 | 0 | 0 | 0 | 1.0000 | tie | 1.0000 (wrong) |
| dual_agent | triple_agent/round0000 | 43 | 986 | 0 | 0 | 0 | 1.0000 | tie | 1.0000 (wrong) |
| dual_agent | triple_agent/round0000 | 44 | 986 | 0 | 0 | 0 | 1.0000 | tie | 1.0000 (wrong) |
| gan/final | gan/round0000 | 42 | 973 | 13 | 156 | 160 | 0.8660 | b>a | 1.0000 (wrong) |
| gan/final | gan/round0000 | 43 | 975 | 11 | 187 | 138 | 0.0077 | a>b | 0.0069 (wrong) |
| gan/final | gan/round0000 | 44 | 970 | 16 | 175 | 153 | 0.2462 | a>b | 0.2519 (wrong) |
| gan/final | gan/round0010 | 42 | 977 | 9 | 155 | 166 | 0.5768 | b>a | 0.6193 (wrong) |
| gan/final | gan/round0050 | 42 | 974 | 12 | 158 | 154 | 0.8652 | a>b | 0.6965 (wrong) |
| gan/final | gan/round0200 | 42 | 969 | 17 | 168 | 149 | 0.3120 | a>b | 0.1103 (wrong) |
| gan/final | rag | 42 | 968 | 18 | 174 | 173 | 1.0000 | a>b | 0.7916 (wrong) |
| gan/final | rag | 43 | 964 | 22 | 175 | 133 | 0.0193 | a>b | 0.0061 (wrong) |
| gan/final | rag | 44 | 960 | 26 | 185 | 172 | 0.5254 | a>b | 0.2559 (wrong) |
| gan/final | sft | 42 | 982 | 4 | 176 | 229 | 0.0097 | b>a | 0.0074 (wrong) |
| gan/final | sft | 43 | 981 | 5 | 204 | 220 | 0.4664 | b>a | 0.3584 (wrong) |
| gan/final | sft | 44 | 980 | 6 | 176 | 226 | 0.0144 | b>a | 0.0074 (wrong) |
| gan/final | triple_agent/final | 42 | 980 | 6 | 255 | 181 | 0.0005 | a>b | 0.0006 (wrong) |
| gan/final | triple_agent/final | 43 | 980 | 6 | 239 | 192 | 0.0266 | a>b | 0.0348 (wrong) |
| gan/final | triple_agent/final | 44 | 980 | 6 | 260 | 220 | 0.0750 | a>b | 0.1013 (wrong) |
| gan/final | triple_agent/round0000 | 42 | 982 | 4 | 300 | 216 | 0.0003 | a>b | 0.0004 (wrong) |
| gan/final | triple_agent/round0000 | 43 | 981 | 5 | 278 | 196 | 0.0002 | a>b | 0.0003 (wrong) |
| gan/final | triple_agent/round0000 | 44 | 980 | 6 | 284 | 196 | 0.0001 | a>b | 0.0001 (wrong) |
| gan/round0000 | gan/round0010 | 42 | 972 | 14 | 165 | 172 | 0.7438 | b>a | 0.6286 (wrong) |
| gan/round0000 | gan/round0050 | 42 | 969 | 17 | 176 | 165 | 0.5882 | a>b | 0.7091 (wrong) |
| gan/round0000 | gan/round0200 | 42 | 965 | 21 | 175 | 147 | 0.1323 | a>b | 0.1135 (wrong) |
| gan/round0000 | rag | 42 | 964 | 22 | 175 | 173 | 0.9573 | a>b | 0.7928 (wrong) |
| gan/round0000 | rag | 43 | 964 | 22 | 165 | 170 | 0.8270 | b>a | 1.0000 (wrong) |
| gan/round0000 | rag | 44 | 955 | 31 | 162 | 169 | 0.7416 | b>a | 1.0000 (wrong) |
| gan/round0000 | sft | 42 | 977 | 9 | 205 | 252 | 0.0313 | b>a | 0.0122 (wrong) |
| gan/round0000 | sft | 43 | 980 | 6 | 203 | 267 | 0.0036 | b>a | 0.0015 (wrong) |
| gan/round0000 | sft | 44 | 976 | 10 | 189 | 259 | 0.0011 | b>a | 0.0004 (wrong) |
| gan/round0000 | triple_agent/final | 42 | 975 | 11 | 257 | 179 | 0.0002 | a>b | 0.0006 (wrong) |
| gan/round0000 | triple_agent/final | 43 | 979 | 7 | 195 | 198 | 0.9197 | b>a | 0.8409 (wrong) |
| gan/round0000 | triple_agent/final | 44 | 976 | 10 | 239 | 219 | 0.3747 | a>b | 0.5153 (wrong) |
| gan/round0000 | triple_agent/round0000 | 42 | 977 | 9 | 319 | 233 | 0.0003 | a>b | 0.0007 (wrong) |
| gan/round0000 | triple_agent/round0000 | 43 | 980 | 6 | 299 | 267 | 0.1925 | a>b | 0.2236 (wrong) |
| gan/round0000 | triple_agent/round0000 | 44 | 976 | 10 | 307 | 238 | 0.0035 | a>b | 0.0082 (wrong) |
| gan/round0010 | gan/round0050 | 42 | 973 | 13 | 158 | 144 | 0.4545 | a>b | 0.3358 (wrong) |
| gan/round0010 | gan/round0200 | 42 | 968 | 18 | 161 | 126 | 0.0446 | a>b | 0.0237 (wrong) |
| gan/round0010 | rag | 42 | 967 | 19 | 184 | 176 | 0.7122 | a>b | 0.4392 (wrong) |
| gan/round0010 | sft | 42 | 981 | 5 | 192 | 233 | 0.0522 | b>a | 0.0335 (wrong) |
| gan/round0010 | triple_agent/final | 42 | 979 | 7 | 258 | 174 | 0.0001 | a>b | 0.0001 (wrong) |
| gan/round0010 | triple_agent/round0000 | 42 | 981 | 5 | 287 | 194 | 0.0000 | a>b | 0.0000 (wrong) |
| gan/round0050 | gan/round0200 | 42 | 966 | 20 | 153 | 134 | 0.2880 | a>b | 0.2268 (wrong) |
| gan/round0050 | rag | 42 | 964 | 22 | 178 | 186 | 0.7137 | b>a | 0.9590 (wrong) |
| gan/round0050 | sft | 42 | 978 | 8 | 185 | 241 | 0.0076 | b>a | 0.0028 (wrong) |
| gan/round0050 | triple_agent/final | 42 | 976 | 10 | 249 | 180 | 0.0010 | a>b | 0.0022 (wrong) |
| gan/round0050 | triple_agent/round0000 | 42 | 978 | 8 | 281 | 203 | 0.0005 | a>b | 0.0011 (wrong) |
| gan/round0200 | rag | 42 | 959 | 27 | 154 | 182 | 0.1406 | b>a | 0.2215 (wrong) |
| gan/round0200 | sft | 42 | 973 | 13 | 173 | 249 | 0.0003 | b>a | 0.0000 (wrong) |
| gan/round0200 | triple_agent/final | 42 | 971 | 15 | 236 | 185 | 0.0147 | a>b | 0.0434 (wrong) |
| gan/round0200 | triple_agent/round0000 | 42 | 973 | 13 | 257 | 201 | 0.0101 | a>b | 0.0200 (wrong) |
| rag | sft | 42 | 972 | 14 | 186 | 242 | 0.0078 | b>a | 0.0039 (wrong) |
| rag | sft | 43 | 969 | 17 | 190 | 250 | 0.0049 | b>a | 0.0011 (wrong) |
| rag | sft | 44 | 965 | 21 | 171 | 235 | 0.0017 | b>a | 0.0002 (wrong) |
| rag | triple_agent/final | 42 | 970 | 16 | 259 | 184 | 0.0004 | a>b | 0.0019 (wrong) |
| rag | triple_agent/final | 43 | 968 | 18 | 213 | 209 | 0.8839 | a>b | 0.8476 (wrong) |
| rag | triple_agent/final | 44 | 965 | 21 | 238 | 211 | 0.2198 | a>b | 0.5453 (wrong) |
| rag | triple_agent/round0000 | 42 | 972 | 14 | 275 | 191 | 0.0001 | a>b | 0.0007 (wrong) |
| rag | triple_agent/round0000 | 43 | 969 | 17 | 254 | 216 | 0.0878 | a>b | 0.1846 (wrong) |
| rag | triple_agent/round0000 | 44 | 965 | 21 | 264 | 190 | 0.0006 | a>b | 0.0047 (wrong) |
| sft | triple_agent/final | 42 | 984 | 2 | 306 | 179 | 0.0000 | a>b | 0.0000 (wrong) |
| sft | triple_agent/final | 43 | 985 | 1 | 270 | 206 | 0.0038 | a>b | 0.0033 (wrong) |
| sft | triple_agent/final | 44 | 986 | 0 | 286 | 194 | 0.0000 | a>b | 0.0000 (wrong) |
| sft | triple_agent/round0000 | 42 | 986 | 0 | 328 | 192 | 0.0000 | a>b | 0.0000 (wrong) |
| sft | triple_agent/round0000 | 43 | 986 | 0 | 320 | 220 | 0.0000 | a>b | 0.0000 (wrong) |
| sft | triple_agent/round0000 | 44 | 986 | 0 | 329 | 189 | 0.0000 | a>b | 0.0000 (wrong) |
| triple_agent/final | triple_agent/round0000 | 42 | 984 | 2 | 237 | 227 | 0.6761 | a>b | 0.7458 (wrong) |
| triple_agent/final | triple_agent/round0000 | 43 | 985 | 1 | 259 | 223 | 0.1108 | a>b | 0.1218 (wrong) |
| triple_agent/final | triple_agent/round0000 | 44 | 986 | 0 | 250 | 202 | 0.0269 | a>b | 0.0269 (wrong) |

## Seed rule (Section 6.6): p_max over seeds, direction in every seed, alpha = 0.05; p_holm within named families

| a | b | seeds | direction | p_max | p_holm | significant | p_max (sensitivity) | significant (sensitivity) |
|---|---|---|---|---|---|---|---|---|
| baseline | dual_agent | 42,43,44 | a>b | 0.1925 | 0.3850 (family 1) | False | 0.2236 (wrong) | False |
| baseline | gan/final | 42,43,44 | mixed | 0.8660 | 0.8660 (family 0) | False | 1.0000 (wrong) | False |
| baseline | gan/round0000 | 42,43,44 | tie | 1.0000 | — | False | 1.0000 (wrong) | False |
| baseline | gan/round0010 | 42 | b>a | 0.7438 | — | False | 0.6286 (wrong) | False |
| baseline | gan/round0050 | 42 | a>b | 0.5882 | — | False | 0.7091 (wrong) | False |
| baseline | gan/round0200 | 42 | a>b | 0.1323 | — | False | 0.1135 (wrong) | False |
| baseline | rag | 42,43,44 | mixed | 0.9573 | 0.9573 (family 1) | False | 1.0000 (wrong) | False |
| baseline | sft | 42,43,44 | b>a | 0.0313 | 0.0939 (family 1) | True | 0.0122 (wrong) | True |
| baseline | triple_agent/final | 42,43,44 | mixed | 0.9197 | — | False | 0.8409 (wrong) | False |
| baseline | triple_agent/round0000 | 42,43,44 | a>b | 0.1925 | — | False | 0.2236 (wrong) | False |
| dual_agent | gan/final | 42,43,44 | b>a | 0.0003 | 0.0005 (family 0) | True | 0.0004 (wrong) | True |
| dual_agent | gan/round0000 | 42,43,44 | b>a | 0.1925 | — | False | 0.2236 (wrong) | False |
| dual_agent | gan/round0010 | 42 | b>a | 0.0000 | — | True | 0.0000 (wrong) | True |
| dual_agent | gan/round0050 | 42 | b>a | 0.0005 | — | True | 0.0011 (wrong) | True |
| dual_agent | gan/round0200 | 42 | b>a | 0.0101 | — | True | 0.0200 (wrong) | True |
| dual_agent | rag | 42,43,44 | b>a | 0.0878 | — | False | 0.1846 (wrong) | False |
| dual_agent | sft | 42,43,44 | b>a | 0.0000 | — | True | 0.0000 (wrong) | True |
| dual_agent | triple_agent/final | 42,43,44 | b>a | 0.6761 | — | False | 0.7458 (wrong) | False |
| dual_agent | triple_agent/round0000 | 42,43,44 | tie | 1.0000 | — | False | 1.0000 (wrong) | False |
| gan/final | gan/round0000 | 42,43,44 | mixed | 0.8660 | — | False | 1.0000 (wrong) | False |
| gan/final | gan/round0010 | 42 | b>a | 0.5768 | — | False | 0.6193 (wrong) | False |
| gan/final | gan/round0050 | 42 | a>b | 0.8652 | — | False | 0.6965 (wrong) | False |
| gan/final | gan/round0200 | 42 | a>b | 0.3120 | — | False | 0.1103 (wrong) | False |
| gan/final | rag | 42,43,44 | a>b | 1.0000 | — | False | 0.7916 (wrong) | False |
| gan/final | sft | 42,43,44 | b>a | 0.4664 | — | False | 0.3584 (wrong) | False |
| gan/final | triple_agent/final | 42,43,44 | a>b | 0.0750 | 0.0750 (family 2) | False | 0.1013 (wrong) | False |
| gan/final | triple_agent/round0000 | 42,43,44 | a>b | 0.0003 | — | True | 0.0004 (wrong) | True |
| gan/round0000 | gan/round0010 | 42 | b>a | 0.7438 | — | False | 0.6286 (wrong) | False |
| gan/round0000 | gan/round0050 | 42 | a>b | 0.5882 | — | False | 0.7091 (wrong) | False |
| gan/round0000 | gan/round0200 | 42 | a>b | 0.1323 | — | False | 0.1135 (wrong) | False |
| gan/round0000 | rag | 42,43,44 | mixed | 0.9573 | — | False | 1.0000 (wrong) | False |
| gan/round0000 | sft | 42,43,44 | b>a | 0.0313 | — | True | 0.0122 (wrong) | True |
| gan/round0000 | triple_agent/final | 42,43,44 | mixed | 0.9197 | — | False | 0.8409 (wrong) | False |
| gan/round0000 | triple_agent/round0000 | 42,43,44 | a>b | 0.1925 | — | False | 0.2236 (wrong) | False |
| gan/round0010 | gan/round0050 | 42 | a>b | 0.4545 | — | False | 0.3358 (wrong) | False |
| gan/round0010 | gan/round0200 | 42 | a>b | 0.0446 | — | True | 0.0237 (wrong) | True |
| gan/round0010 | rag | 42 | a>b | 0.7122 | — | False | 0.4392 (wrong) | False |
| gan/round0010 | sft | 42 | b>a | 0.0522 | — | False | 0.0335 (wrong) | True |
| gan/round0010 | triple_agent/final | 42 | a>b | 0.0001 | — | True | 0.0001 (wrong) | True |
| gan/round0010 | triple_agent/round0000 | 42 | a>b | 0.0000 | — | True | 0.0000 (wrong) | True |
| gan/round0050 | gan/round0200 | 42 | a>b | 0.2880 | — | False | 0.2268 (wrong) | False |
| gan/round0050 | rag | 42 | b>a | 0.7137 | — | False | 0.9590 (wrong) | False |
| gan/round0050 | sft | 42 | b>a | 0.0076 | — | True | 0.0028 (wrong) | True |
| gan/round0050 | triple_agent/final | 42 | a>b | 0.0010 | — | True | 0.0022 (wrong) | True |
| gan/round0050 | triple_agent/round0000 | 42 | a>b | 0.0005 | — | True | 0.0011 (wrong) | True |
| gan/round0200 | rag | 42 | b>a | 0.1406 | — | False | 0.2215 (wrong) | False |
| gan/round0200 | sft | 42 | b>a | 0.0003 | — | True | 0.0000 (wrong) | True |
| gan/round0200 | triple_agent/final | 42 | a>b | 0.0147 | — | True | 0.0434 (wrong) | True |
| gan/round0200 | triple_agent/round0000 | 42 | a>b | 0.0101 | — | True | 0.0200 (wrong) | True |
| rag | sft | 42,43,44 | b>a | 0.0078 | — | True | 0.0039 (wrong) | True |
| rag | triple_agent/final | 42,43,44 | a>b | 0.8839 | — | False | 0.8476 (wrong) | False |
| rag | triple_agent/round0000 | 42,43,44 | a>b | 0.0878 | — | False | 0.1846 (wrong) | False |
| sft | triple_agent/final | 42,43,44 | a>b | 0.0038 | — | True | 0.0033 (wrong) | True |
| sft | triple_agent/round0000 | 42,43,44 | a>b | 0.0000 | — | True | 0.0000 (wrong) | True |
| triple_agent/final | triple_agent/round0000 | 42,43,44 | a>b | 0.6761 | — | False | 0.7458 (wrong) | False |

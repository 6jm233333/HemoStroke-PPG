# Benchmark Results

These benchmark values are the study-level reference results associated with this code release. They are not recomputed by the unit tests because full numerical reproduction requires restricted clinical waveform and EHR data.

## Cohorts

| Cohort | Source population | Stroke subset with eligible pre-onset PPG |
|---|---:|---:|
| MIMIC-III | 38,548 | 176 |
| MC-MED | 118,385 | 158 |

## Effective Window-Level Units

| Horizon | MIMIC train total/normal/warn | MIMIC val total/normal/warn | MIMIC held-out total/normal/warn | MC-MED total/normal/warn |
|---:|---:|---:|---:|---:|
| 240 min | 20,162 / 6,862 / 13,300 | 2,746 / 891 / 1,855 | 7,205 / 2,420 / 4,785 |  737 / 232 / 505 |
| 300 min | 20,306 / 4,433 / 15,873 | 2,778 / 599 / 2,179 | 7,252 / 1,521 / 5,731 | 748 / 173 / 575 |
| 360 min | 20,510 / 2,349 / 18,161 | 2,778 / 309 / 2,469 | 7,320 / 747 / 6,573 | 752 / 118 / 634 |

## Predictive Performance

| Horizon | Cohort | Model | Accuracy | Recall | Precision | F1 | F2 | AUC |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 240 | MIMIC-III | ResNet-1D | 0.6654 +/- 0.0047 | 0.9833 +/- 0.0106 | 0.6681 +/- 0.0041 | 0.7956 +/- 0.0027 | 0.8985 +/- 0.0062 | 0.6525 +/- 0.0530 |
| 240 | MIMIC-III | LSTM | 0.6645 +/- 0.0061 | 0.9047 +/- 0.0888 | 0.6907 +/- 0.0277 | 0.7801 +/- 0.0161 | 0.8492 +/- 0.0546 | 0.6394 +/- 0.0170 |
| 240 | MC-MED | ResNet-1D | 0.8976 +/- 0.0350 | 0.9341 +/- 0.0413 | 0.9179 +/- 0.0020 | 0.9256 +/- 0.0211 | 0.9306 +/- 0.0332 | 0.5595 +/- 0.1020 |
| 240 | MC-MED | LSTM | 0.7719 +/- 0.0451 | 0.8188 +/- 0.0472 | 0.8361 +/- 0.0399 | 0.7456 +/- 0.0520 | 0.7471 +/- 0.0530 | 0.6129 +/- 0.0491 |
| 300 | MIMIC-III | ResNet-1D | 0.7860 +/- 0.0129 | 0.9647 +/- 0.0360 | 0.8028 +/- 0.0125 | 0.8759 +/- 0.0105 | 0.9269 +/- 0.0243 | 0.6924 +/- 0.0668 |
| 300 | MIMIC-III | LSTM | 0.7695 +/- 0.0055 | 0.9398 +/- 0.0143 | 0.8015 +/- 0.0039 | 0.8651 +/- 0.0047 | 0.9085 +/- 0.0102 | 0.6649 +/- 0.0115 |
| 300 | MC-MED | ResNet-1D | 0.9394 +/- 0.0273 | 0.9401 +/- 0.0294 | 0.9802 +/- 0.0023 | 0.9595 +/- 0.0151 | 0.9478 +/- 0.0238 | 0.5847 +/- 0.1560 |
| 300 | MC-MED | LSTM | 0.9201 +/- 0.0487 | 0.9367 +/- 0.0545 | 0.9809 +/- 0.0039 | 0.9576 +/- 0.0275 | 0.9447 +/- 0.0441 | 0.6109 +/- 0.0729 |
| 360 | MIMIC-III | ResNet-1D | 0.8880 +/- 0.0028 | 0.9981 +/- 0.0025 | 0.8894 +/- 0.0013 | 0.9406 +/- 0.0015 | 0.9743 +/- 0.0020 | 0.7492 +/- 0.1147 |
| 360 | MIMIC-III | LSTM | 0.8794 +/- 0.0067 | 0.9697 +/- 0.0123 | 0.9021 +/- 0.0073 | 0.9346 +/- 0.0040 | 0.9553 +/- 0.0085 | 0.7346 +/- 0.0469 |
| 360 | MC-MED | ResNet-1D | 0.9814 +/- 0.0192 | 0.9804 +/- 0.0054 | 0.9975 +/- 0.0008 | 0.9888 +/- 0.0025 | 0.9837 +/- 0.0042 | 0.7079 +/- 0.1248 |
| 360 | MC-MED | LSTM | 0.9715 +/- 0.0171 | 0.9732 +/- 0.0172 | 0.9981 +/- 0.0005 | 0.9854 +/- 0.0088 | 0.9780 +/- 0.0139 | 0.7827 +/- 0.0284 |

## False-Alert Burden

| Horizon | MIMIC FPR | MIMIC TPR | MIMIC Patient+ | MC-MED FPR | MC-MED TPR | MC-MED Patient+ |
|---:|---:|---:|---:|---:|---:|---:|
| 240 | 0.167 | 0.983 | 0.061 | 0.166 | 0.934 | 0.030 |
| 300 | 0.288 | 0.965 | 0.145 | 0.197 | 0.940 | 0.038 |
| 360 | 0.294 | 0.998 | 0.174 | 0.137 | 0.980 | 0.018 |

## Robustness Checks

- Relative-feature ablation did not collapse performance, indicating that performance is not driven only by baseline normalization.
- Onset-anchor perturbation at +/-15, +/-30, and +/-60 minutes preserved non-trivial internal performance.
- Signal-quality audit reported MIMIC-III SQI 0.87 [0.79--0.93] and MC-MED SQI 0.93 [0.86--0.96].
- SHAP and trajectory analyses highlighted systolic timing and amplitude-related morphology, especially `T_sp_Rel`, `T_sp`, and `A_sp_Rel`.

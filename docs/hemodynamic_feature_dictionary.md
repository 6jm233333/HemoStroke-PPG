# Hemodynamic Feature Dictionary

This document provides the human-readable definition of the 17 photoplethysmography-derived hemodynamic features used in the study.

These features are derived from the raw PPG signal, the velocity plethysmogram (VPG), and the acceleration plethysmogram (APG). They are organized into four groups: time-domain and morphological features, derivative and normalized features, short-term variability features, and relative deviation features.

## I. Time-Domain and Morphological Features

| Symbol | Definition | Physiological significance |
|---|---|---|
| `T_sp` | Time to systolic peak. Time elapsed from pulse onset to the systolic peak. | Reflects the velocity of blood ejection and wave transmission. It is related to arterial compliance and may shorten with arterial stiffening. |
| `SI` | Stiffness index. Ratio of pulse amplitude to systolic time. | Represents the average slope of the systolic upstroke. A higher value may indicate increased arterial stiffness. |
| `A_off` | Offset amplitude. Signal amplitude at the pulse offset point. | Reflects vascular recoil at end-diastole and may relate to vessel elasticity and venous return. |
| `T_sys/T_dia` | Systolic-diastolic ratio. Ratio of systolic duration to diastolic duration. | Describes the temporal balance of the cardiac cycle. Abnormal balance may reflect autonomic dysfunction or altered vascular resistance. |

## II. Derivative and Normalized Features

| Symbol | Definition | Physiological significance |
|---|---|---|
| `T_u_Tpi` | Normalized time to VPG peak. Time to maximum velocity, normalized by pulse interval. | Reflects the relative duration of the rapid ejection phase and may relate to vascular impedance and flow inertia. |
| `T_b_Tpi` | Normalized time to APG b-wave. Time to the b-wave in the second derivative, normalized by pulse interval. | Sensitive to vascular aging and early peripheral wave reflection changes. |
| `T_v` | Time to VPG valley. Time to the post-systolic VPG inflection point. | Correlates with the timing of reflected wave return and central-peripheral hemodynamic coupling. |
| `T_u_Ta_Tpi` | VPG-APG peak delay ratio. Temporal difference between peak velocity and acceleration. | Describes decoupling between peak velocity and acceleration and may reflect neurovascular mismatch. |

## III. Short-Term Variability Features

| Symbol | Definition | Physiological significance |
|---|---|---|
| `CV_T_pi` | Pulse interval variability. Coefficient of variation of pulse interval. | Serves as an ultra-short-term heart-rate-variability surrogate. Reduced variability may indicate autonomic imbalance. |
| `CV_PA` | Amplitude variability. Coefficient of variation of pulse amplitudes. | Reflects the capacity of sympathetic regulation over peripheral vascular tone. Reduced variability may indicate rigid regulation. |

## IV. Relative Deviation Features

| Symbol | Definition | Physiological significance |
|---|---|---|
| `T_sp_Rel` | Relative time to systolic peak. Deviation of current `T_sp` from the individual baseline. | Captures progressive drift in wave transmission time and reduces inter-subject baseline variability. |
| `A_sp_Rel` | Relative systolic amplitude. Deviation of systolic peak amplitude from the individual baseline. | Sensitive to acute changes in vascular compliance and stroke-volume-related pulse morphology. |
| `SI_Rel` | Relative stiffness index. Deviation of stiffness index from the individual baseline. | Tracks patient-specific change in arterial stiffness or compliance degradation. |
| `DSI_Rel` | Relative dynamic stability index. Deviation of dynamic stability from the individual baseline. | Quantifies shifts in waveform morphology and may reflect loss of hemodynamic homeostasis. |
| `T_c_Rel` | Relative APG c-wave timing. Shift in late-systolic c-wave timing from baseline. | Relates to late-systolic recoil and may indicate altered arterial wall elasticity. |
| `A_off_Rel` | Relative offset amplitude. Deviation in end-diastolic amplitude from baseline. | Associated with changes in venous return and diastolic regulation. |
| `A_on_Rel` | Relative onset amplitude. Deviation in pulse-onset nadir from baseline. | Sensitive to peripheral perfusion and microcirculatory abnormalities. |

## Notes

The relative deviation features are computed with respect to a subject-specific baseline to reduce inter-patient heterogeneity and capture patient-level hemodynamic trajectories.

This document is provided as an online feature dictionary for reproducibility. It is not a substitute for the paper, and restricted clinical data are not redistributed in this repository.

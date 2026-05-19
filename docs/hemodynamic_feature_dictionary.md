# Hemodynamic Feature Dictionary

This document provides the human-readable definition of the 17 photoplethysmography-derived hemodynamic features used in the study.

These features are derived from the raw photoplethysmography (PPG) signal, the velocity plethysmogram (VPG), and the acceleration plethysmogram (APG). They are organized into four groups: time-domain and morphological features, derivative and normalized features, short-term variability features, and relative deviation features.

## I. Time-Domain and Morphological Features

| Symbol | Definition | Physiological significance |
|---|---|---|
| $T_{sp}$ | Time to systolic peak. Time elapsed from pulse onset to the systolic peak. | Reflects the velocity of blood ejection and wave transmission. It is related to arterial compliance and may shorten with arterial stiffening. |
| $SI$ | Stiffness index. A stiffness-related index derived from pulse morphology. | Represents systolic upstroke characteristics and arterial stiffness. A higher value may indicate increased arterial stiffness. |
| $A_{off}$ | Offset amplitude. Signal amplitude at the pulse offset point. | Reflects vascular recoil at end-diastole and may relate to vessel elasticity and venous return. |
| $T_{sys}/T_{dia}$ | Systolic-diastolic ratio. Ratio of systolic duration to diastolic duration. | Describes the temporal balance of the cardiac cycle. Abnormal balance may reflect autonomic dysfunction or altered vascular resistance. |

## II. Derivative and Normalized Features

| Symbol | Definition | Physiological significance |
|---|---|---|
| $T_{u_{T_{p_i}}}$ | Normalized time to VPG peak. Time to the VPG $u$ point normalized by pulse interval $T_{p_i}$. | Reflects the relative duration of the rapid ejection phase and may relate to vascular impedance and flow inertia. |
| $T_{b_{T_{p_i}}}$ | Normalized time to APG b-wave. Time to the APG $b$ wave normalized by pulse interval $T_{p_i}$. | Sensitive to vascular aging and early peripheral wave reflection changes. |
| $T_v$ | Time to VPG valley. Time to the post-systolic VPG inflection point. | Correlates with the timing of reflected wave return and central-peripheral hemodynamic coupling. |
| $T_{u_{T_a,T_{p_i}}}$ | VPG-APG timing relation. Timing relation between the VPG $u$ point, APG timing $T_a$, and pulse interval $T_{p_i}$. | Describes the coupling between velocity and acceleration components of the PPG waveform and may reflect neurovascular mismatch. |

## III. Short-Term Variability Features

| Symbol | Definition | Physiological significance |
|---|---|---|
| $CV_{T,p_i}$ | Pulse interval variability. Coefficient of variation of pulse interval $T_{p_i}$. | Serves as an ultra-short-term heart-rate-variability surrogate. Reduced variability may indicate autonomic imbalance. |
| $CV_{PA}$ | Pulse amplitude variability. Coefficient of variation of pulse amplitudes. | Reflects beat-to-beat variability in peripheral pulse amplitude and the capacity of sympathetic regulation over peripheral vascular tone. |

## IV. Relative Deviation Features

| Symbol | Definition | Physiological significance |
|---|---|---|
| $T_{sp,\mathrm{Rel}}$ | Relative time to systolic peak. Deviation of current $T_{sp}$ from the individual baseline. | Captures progressive drift in wave transmission time and reduces inter-subject baseline variability. |
| $A_{sp,\mathrm{Rel}}$ | Relative systolic amplitude. Deviation of systolic peak amplitude $A_{sp}$ from the individual baseline. | Sensitive to acute changes in vascular compliance and stroke-volume-related pulse morphology. |
| $SI_{\mathrm{Rel}}$ | Relative stiffness index. Deviation of $SI$ from the individual baseline. | Tracks patient-specific change in arterial stiffness or compliance degradation. |
| $DSI_{\mathrm{Rel}}$ | Relative dynamic stability index. Deviation of dynamic stability index from the individual baseline. | Quantifies shifts in waveform morphology and may reflect loss of hemodynamic homeostasis. |
| $T_{c,\mathrm{Rel}}$ | Relative APG c-wave timing. Shift in late-systolic c-wave timing from baseline. | Relates to late-systolic recoil and may indicate altered arterial wall elasticity. |
| $A_{off,\mathrm{Rel}}$ | Relative offset amplitude. Deviation in end-diastolic amplitude from baseline. | Associated with changes in venous return and diastolic regulation. |
| $A_{on,\mathrm{Rel}}$ | Relative onset amplitude. Deviation in pulse-onset nadir from baseline. | Sensitive to peripheral perfusion and microcirculatory abnormalities. |

## Relative Deviation Formula

For a feature $x(t)$, the relative deviation feature is computed with respect to a subject-specific baseline:

$$
x_{\mathrm{Rel}}(t)=\frac{x(t)-\mu_{\mathrm{base}}}{|\mu_{\mathrm{base}}|}
$$

where $\mu_{\mathrm{base}}$ denotes the mean value of the corresponding feature during the subject-specific baseline window.

## Notes

The relative deviation features are computed with respect to a subject-specific baseline to reduce inter-patient heterogeneity and capture patient-level hemodynamic trajectories.

This document is provided as an online feature dictionary for reproducibility. It is not a substitute for the paper, and restricted clinical data are not redistributed in this repository.

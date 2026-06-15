# Interpretation & treatment evidence (what the biofeedback means, and what to do)

The cited evidence behind `analysis/interpretation.py` (the plain-language "what it means /
what helps / how to track change" the report shows under each flag) and the task analysis in
`analysis/tasks.py`. Effect sizes and evidence-strength notes are kept so nothing is over-sold.

## Honest framing (threads through all of it)
- Movement data is best-validated for **assessment and targeting**, not as a standalone cure.
- The **underlying therapies** (strengthening, mobilization, vestibular rehab, perturbation
  training, cueing, aerobic exercise) have the strongest evidence; **biofeedback as an add-on**
  is more variable and often short-term / not sham-controlled.
- Flags are decision-support, not a diagnosis. No muscle strength/structure claims from kinematics.

## Gait retraining & real-time biofeedback (efficacy)
- **Running cadence +7.5–10%** cuts vertical loading rate ~16% and reduced PFP pain (RCT: VAS
  −2.9 cm at 6 mo), retained 1–3 mo. ([PMC10781021](https://pmc.ncbi.nlm.nih.gov/articles/PMC10781021/), [PMC7450991](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7450991/))
- **Tibial-accelerometer feedback**: loading rate −30%, retained at 1 mo (Crowell & Davis, [JOSPT 2010](https://www.jospt.org/doi/10.2519/jospt.2010.3166)).
- **Reducing hip adduction / pelvic drop** (mirror cueing) reduced PFP/ITBS-linked mechanics, held 1–3 mo (Willy/Noehren/Davis).
- **Knee OA — reduce KAM**: lateral trunk lean (dose-dependent KAM↓; [Simic 2012](https://acrjournals.onlinelibrary.wiley.com/doi/10.1002/acr.21724)); toe-in haptic feedback **KAM −20%, WOMAC pain −29%**, retained 1 mo ([Shull 2013](https://med.stanford.edu/content/dam/sm/ortho/documents/humanperformance/publications/Shull_JOB_2013.pdf)). **OpenCap predicts KAM at r²=0.80** → clinic-deployable ([OpenCap 2023](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462), [OA app 2025](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1674133/full)).
- **Stroke auditory cueing**: walking velocity Hedges g≈0.98, cadence 0.84, stride 0.76 ([Yoo 2016](https://pubmed.ncbi.nlm.nih.gov/27084833/), [Ghai 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6379377/)); optimal 20–45 min, 3–5×/wk.
- **Parkinson's cueing**: stride length g≈0.48, velocity 0.27; visual cues add stride length specifically; **does NOT reliably abolish freezing** ([Ghai 2017](https://www.nature.com/articles/s41598-017-16232-5), [one-cue-doesn't-fit-all](https://pubmed.ncbi.nlm.nih.gov/37086934/)).
- **Falls — perturbation/reactive balance training**: fall rate ↓ 23–24% (RR 0.71, rate ratio 0.54; [Mansfield 2015](https://pubmed.ncbi.nlm.nih.gov/25524873/)), some reviews ~39–50%.
- *Caveats:* small samples, short follow-up (1–3 mo), weak blinding; KAM/loading-rate are load *surrogates* not hard endpoints.

## Squat & sit-to-stand interpretation (your task)
- **Depth** needs ~95° hip flexion + ~40° ankle DF for a flat-heel deep squat ([PMC7276781](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7276781/)). Bands: ~parallel ≈ 90° knee flexion, deep >100°.
- **Butt wink** = `pelvis_tilt` reversing toward posterior near depth; usual cause **ankle DF limit or hip morphology/FAI**, *not* hamstrings (myth). Injury risk is **load-dependent**, weak/contested at bodyweight ([Prehab Guys](https://theprehabguys.com/pelvic-tilt-and-squat-depth/), [Barbell Rehab](https://barbellrehab.com/stop-fearing-spinal-flexion/)).
- **Dynamic valgus** (rising `hip_adduction`): FPPA **≥10° increase** flags it; MKD ~12.9° vs 6° controls; links to ACL (2–9× female risk) and PFP. Treat hip-abductor strength **+ motor control** — strength reliably cuts **pain** but its effect on the **valgus angle is inconsistent** ([PMC7664395](https://pmc.ncbi.nlm.nih.gov/articles/PMC7664395/), [PMC4424529](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4424529/), [JOSPT 2018](https://www.jospt.org/doi/10.2519/jospt.2018.7365)).
- **Ankle DF restriction**: knee-to-wall needs ≥10–12.5 cm (~1 cm ≈ 3.6°); joint vs muscle limit changes treatment (mobilization vs stretch); 8-wk program gave clinically significant gains ([PMC11527180](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11527180/)).
- **5× sit-to-stand** norms: 60s 11.4s, 70s 12.6s, 80s 14.8s; **>12s** assess, **>15s** elevated fall risk ([Physiopedia](https://www.physio-pedia.com/Five_Times_Sit_to_Stand_Test), [APTA 5TSTS](https://www.neuropt.org/docs/default-source/cpgs/core-outcome-measures/5tsts-pocket-guide-v2-proof9-(2)38db36a5390366a68a96ff00001fc240.pdf)).
- **Decision logic:** classify each fault as **mobility vs strength vs motor control**, treat the *limiter*, re-test the same metric.

## Monitoring: what change is real (MCID/MDC) & how the data changes decisions
- **Gait speed MCID ~0.10–0.17 m/s** (Bohannon 2014, [JECP](https://onlinelibrary.wiley.com/doi/10.1111/jep.12158)); poststroke "substantial" ≈ 0.10–0.16 m/s; speed is the "sixth vital sign" (each −0.1 m/s ≈ 10–12% more risk; <0.8 m/s flags risk).
- **GDI**: 100 = typical, each 10 pts = 1 SD; CP smallest detectable change ~11–18%; **post-stroke MDC is large (~7.5–9.4)** → blunt there ([PubMed 26043670](https://pubmed.ncbi.nlm.nih.gov/26043670/), [28073084](https://pubmed.ncbi.nlm.nih.gov/28073084/)).
- **Gait analysis changes CP surgical plans in ~42–95% of cases (commonly ~50–70%)**, usually *less* surgery ([Lofterød 2007: 42/60](https://pubmed.ncbi.nlm.nih.gov/17453395/), [EJPN review](https://pubmed.ncbi.nlm.nih.gov/36563467/)).
- **Markerless reliability**: spatiotemporal (speed/cadence/stride) most trustworthy; **between-session joint-angle MDC often >5°**; average ≥3 trials, standardize setup; don't over-read frontal/transverse or high-speed ([J Sports Sci 2024](https://www.tandfonline.com/doi/full/10.1080/02640414.2024.2415233), [Sensors 2026](https://doi.org/10.3390/s26041234)).
- **Showing patients their movement** improves engagement/adherence (strong); superior *clinical outcomes* suggested but not yet proven by large RCTs.

## Neuro / vestibular / concussion / balance
- **Vestibular rehab** works (Cochrane OR 2.67, 1.85–3.86; DHI ↓5–42 pts). Audio/biofeedback *augmentation* not clearly superior ([Cochrane](https://www.cochranelibrary.com/cdsr/doi/10.1002/14651858.CD005397.pub4/full)).
- **Concussion**: dual-task gait deficits persist ~2 mo and predict re-injury → good *gating* tool; but dual-task *training* is NOT superior to single-task. Strongest treatment = **treadmill-test-individualized sub-symptom aerobic exercise** (faster recovery; [Buffalo](https://www.buffalo.edu/news/releases/2021/09/040.html)).
- **Falls**: perturbation/reactive training RR 0.71; **vibrotactile sway feedback** helps *while worn* but weak durable carry-over ([Mansfield 2015](https://pubmed.ncbi.nlm.nih.gov/25524873/)).
- **Parkinson's**: cueing helps the deficit it matches; **arm-swing asymmetry is a dopa-responsive biomarker** for tracking therapy.
- **Stroke**: weight-bearing/visual symmetry biofeedback helps in **subacute**; less consistent in chronic (esp. added to treadmill).
- **Telerehab/home biofeedback**: broadly **non-inferior** (balance SMD ~0.25); adherence is the limiter.

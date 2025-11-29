# Adult-Infant Neural Coupling Analysis

This repository contains the analysis code for the research paper:

**Adult-infant neural coupling mediates infants’ selection of socially-relevant stimuli for learning across cultures**

Wei Zhang<sup>1,2</sup>, Kaili Clackson<sup>1</sup>, Stanimira Georgieva<sup>1</sup>, Lorena Santamaria<sup>1</sup>, Vanessa Reindl<sup>3,4</sup>, Valdas Noreika<sup>5</sup>, Nicholas Darby<sup>6</sup>, Vaka Valsdottir<sup>6</sup>, Priyadharshini Santhanakrishnan<sup>1</sup>, & Victoria Leong<sup>1</sup>

<sup>1</sup> Early Mental Potential and Wellbeing Research (EMPOWER) Centre, Nanyang Technological University, Singapore
<sup>2</sup> Cognitive Neuroimaging Centre, Nanyang Technological University, Singapore
<sup>3</sup> Division of Psychology, Nanyang Technological University, Singapore
<sup>4</sup> Section Child Neuropsychology, Department of Child and Adolescent Psychiatry, Psychosomatics and Psychotherapy, Uniklinik RWTH Aachen, Germany
<sup>5</sup> Department of Psychology, School of Biological and Chemical Sciences, Queen Mary University of London, London, UK
<sup>6</sup> Department of Psychology, University of Cambridge, UK

## Overview

This repository contains the analytical code used to investigate how adult-infant neural coupling mediates the selection of socially-relevant stimuli for learning in both British and Singaporean infant populations. The analysis focuses on neural synchrony between adult speaker and infants listener through the use of pre-recorded video, and how this relates to statistical learning outcomes.

## Citation

If you use this code in your research, please cite our paper:

```
Zhang, W., Clackson, K., Georgieva, S., Santamaria, L., Reindl, V., Noreika, V., Darby, N., Valsdottir, V., Santhanakrishnan, P., & Leong, V. (2025). Adult-infant neural coupling mediates infants’ selection of socially-relevant stimuli for learning across cultures.
```

## Data Availability

All datasets have been made publicly available through Nanyang Technological University (NTU)'s data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to NTU's open access policy:

1. **Raw EEG data**: https://doi.org/10.21979/N9/BQLIB9
2. **Preprocessed EEG data**: https://doi.org/10.21979/N9/F9N5BE
3. **Behavioral data**: https://doi.org/10.21979/N9/4EBTKT
4. **Video stimuli**: https://doi.org/10.21979/N9/NJ1KJA

This code repository demonstrates the analytical methodology used in our study. The scripts are designed to work with the publicly available data listed above.

## Analysis Pipeline

The analysis consists of the following sequential steps.

### Core Analysis Scripts

1. **Step 01**: Calculate attention measures from EEG segments
2. **Step 02**: Calculate learning scores and attention proportions
3. **Step 03 (Behavior)**: Behavioral analyses (attention, CDI, demographics)
4. **Step 03 (Learning)**: Three-tier hierarchical learning analysis
5. **Step 04**: Calculate GPDC (Generalized Partial Directed Coherence) connectivity
   - Input: Preprocessed EEG data (individual participant files)
   - Output: GPDC matrices for each participant/block/condition
   - Format: `UK_###_PDC.mat`, `SG_###_PDC.mat`

6. **Step 05**: Identify significant connections (NON-CIRCULAR)
   - **NEW**: Surrogate testing for feature selection
   - Input: Step 6 aggregated data + surrogate distributions
   - Method: Real vs. surrogate baseline (NO learning data used)
   - Output: `stronglistfdr5_gpdc_*.mat` files
   - Purpose: Prevents circular analysis in subsequent steps

7. **Step 06**: Aggregate GPDC data into analysis matrix
   - **NEW**: Data integration script
   - Input: Individual GPDC files from Step 4
   - Output: `data_read_surr_gpdc2.mat` (226 obs × 981 variables)
   - Structure: Demographics + 972 connectivity values
   - Used by: Steps 5, 11, 12, 18

8. **Step 07-10**: Additional connectivity analyses
9. **Step 11**: PLS (Partial Least Squares) prediction of learning from connectivity
   - Uses significant connections from Step 5 (non-circular!)
   - Validation: Surrogate testing, cross-validation, bootstrap

10. **Step 12**: Mediation analysis (gaze → connectivity → learning pathway)
    - Exploratory analysis with analytical dependencies acknowledged
    - Validation: Negative controls (II GPDC) + Step 18 convergence

### Model Validation Scripts

11. **Step 16**: BIC-based MVAR model order selection
12. **Step 17**: MVAR model diagnostics (variance explained, stability analysis)
13. **Step 18**: Single-connection validation (non-circular feature selection)
    - **UPDATED**: Fully runnable with LME bootstrap mediation
    - Method: Condition-based selection (gaze effect, NO learning data)
    - Identifies: Adult Fz → Infant F4 (sole significant connection, pFDR=.048)
    - Tests: Full mediation pathway with 1000 bootstrap iterations
    - Result: Convergence with main analysis validates genuine neural pathway
14. **Step 19**: Frequency robustness analysis (delta, theta, alpha bands)
15. **Step 20**: Statistical power and sensitivity analysis
16. **Step 21**: Alternative mediation models (negative controls)
17. **Step 22**: Order effects analysis (within-subjects design validation)

These validation scripts demonstrate rigorous model selection, adequacy checks, and methodological robustness for all major analyses.

## Requirements

The code in this repository requires the following major dependencies:
- MATLAB R2024b (https://www.mathworks.com)
- EEGLAB 2021.1 (https://eeglab.org/)
- eMVAR (http://www.lucafaes.net/emvar.html)

## Acknowledgements

This research is supported by the RIE2025 Human Potential Programme Prenatal/Early Childhood Grant (H22P0M0002), administered by A*STAR. VL is supported by the Ministry of Education, Singapore, under its Academic Research Fund Tier 2 (MOE-T2EP40121-0001) and by a Social Science & Humanities Research Fellowship (MOE2020-SSHR-008).

Special thanks to Eszter and Diarmid Campbell for invaluable practical advice on the use of polarising filters to modulate visibility of the speaker's eyes while still allowing the speaker to read syllable strings from the autocue.

Thanks also to Melis Çetinçelik, Marina Wenzl, Winnie Wee, Ivfy Foong, Teo Kai Xin, Dorcas Keow, Yvonne Chia, Jamie Lee, Sim Jia Yi, Lois Timothy, Arya Bhomick, Nastassja Fischer, Lee Kean Mun for assistance with data collection and analysis.

We thank and acknowledge the Cognitive Neuroimaging Centre, Nanyang Technological University, Singapore for computational resources used in data analysis.

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Copyright (c) 2025 Zhang et al.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the material without restriction, including the rights to use, copy, modify, remix, transform, and build upon the material, provided that:

Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

NonCommercial — You may not use the material for commercial purposes.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

The full license text is available at: https://creativecommons.org/licenses/by-nc/4.0/

## Contact

For any questions regarding this code repository, please contact:

PI: Prof. Victoria Leong  

Early Mental Potential and Wellbeing Research (EMPOWER) Centre (https://www.ntu.edu.sg/empower),  
Nanyang Technological University, Singapore  
Email: victorialeong@ntu.edu.sg

or the author: Dr. Wei Zhang

Early Mental Potential and Wellbeing Research (EMPOWER) Centre (https://www.ntu.edu.sg/empower),  
Nanyang Technological University, Singapore  
Email: wilson.zhangwei@ntu.edu.sg 

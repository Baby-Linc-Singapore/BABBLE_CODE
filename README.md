# Adult-to-Infant Neural Coupling Analysis

This repository contains the analysis code for the research paper:

**Adult-to-Infant neural coupling mediates infants’ selection of socially-relevant stimuli for learning across cultures**

Wei Zhang<sup>1,2</sup>, Kaili Clackson<sup>1</sup>, Stanimira Georgieva<sup>1</sup>, Lorena Santamaria<sup>1</sup>, Vanessa Reindl<sup>3,4</sup>, Valdas Noreika<sup>5</sup>, Nicholas Darby<sup>6</sup>, Vaka Valsdottir<sup>6</sup>, Priyadharshini Santhanakrishnan<sup>1</sup>, & Victoria Leong<sup>1,*</sup>

<sup>1</sup> Early Mental Potential and Wellbeing Research (EMPOWER) Centre, Nanyang Technological University, Singapore
<sup>2</sup> Cognitive Neuroimaging Centre, Nanyang Technological University, Singapore
<sup>3</sup> Division of Psychology, Nanyang Technological University, Singapore
<sup>4</sup> Section Child Neuropsychology, Department of Child and Adolescent Psychiatry, Psychosomatics and Psychotherapy, Uniklinik RWTH Aachen, Germany
<sup>5</sup> Department of Psychology, School of Biological and Chemical Sciences, Queen Mary University of London, London, UK
<sup>6</sup> Department of Psychology, University of Cambridge, UK

## Overview

This repository contains the analytical code used to investigate how Adult-to-Infant neural coupling mediates the selection of socially-relevant stimuli for learning in both British and Singaporean infant populations. The analysis focuses on neural synchrony between adult speaker and infants listener through the use of pre-recorded video, and how this relates to statistical learning outcomes.

## Citation

If you use this code in your research, please cite our paper:

```
Zhang, W., Clackson, K., Georgieva, S., Santamaria, L., Reindl, V., Noreika, V., Darby, N., Valsdottir, V., Santhanakrishnan, P., & Leong, V. (2025). Adult-to-Infant neural coupling mediates infants’ selection of socially-relevant stimuli for learning across cultures.
```

## Data Availability

All datasets have been made publicly available through Nanyang Technological University (NTU)'s data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to NTU's open access policy:

1. **Raw EEG data**: https://doi.org/10.21979/N9/BQLIB9
2. **Preprocessed EEG data**: https://doi.org/10.21979/N9/F9N5BE
3. **Behavioral data**: https://doi.org/10.21979/N9/4EBTKT
4. **Video stimuli**: https://doi.org/10.21979/N9/NJ1KJA

This code repository demonstrates the analytical methodology used in our study. The scripts are designed to work with the publicly available data listed above.

## Analysis Pipeline

The analysis consists of these sequential steps:  
**Step 01**: Calculate attention measures from EEG segments  
**Step 02**: Calculate learning scores and attention proportions  
**Step 03**: Behavioral analyses (attention, CDI, demographics)  
**Step 03a**: Three-tier hierarchical learning analysis  
**Step 04**: Calculate GPDC (Generalized Partial Directed Coherence) connectivity matrices  
**Step 05**: Identify significant connections via surrogate testing (non-circular)  
**Step 05a**: Generate surrogate distributions for GPDC validation  
**Step 06**: Aggregate individual GPDC files into analysis matrix  
**Step 06a**: Read and organize GPDC and surrogate data  
**Step 07**: GPDC statistical testing and visualization  
**Step 08**: Neural entrainment analysis between speech and EEG  
**Step 09**: Generate surrogate data for entrainment analysis  
**Step 10**: Statistical testing of entrainment effects  
**Step 11**: PLS (Partial Least Squares) regression predicting learning from connectivity  
**Step 12**: Mediation analysis (gaze → connectivity → learning pathway)  
**Step 13**: Calculate EEG data rejection ratios  
**Step 14**: Sample size and statistical power estimation  
**Step 15**: Subject-level learning analysis  
**Step 16**: BIC-based MVAR model order selection  
**Step 17**: MVAR model diagnostics (variance explained, stability)  
**Step 18**: Single-connection validation with non-circular feature selection  
**Step 19**: Frequency robustness analysis (delta, theta, alpha bands)  
**Step 20**: Statistical power and sensitivity analysis  
**Step 21**: Alternative mediation models (negative controls)  
**Step 22**: Order effects analysis (within-subjects design validation)

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

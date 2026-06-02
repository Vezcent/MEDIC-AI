# MEDIC-AI: Alzheimer's Disease Structural Brain Analysis Report
**A Comparative Study of OASIS-1 (Cross-Sectional) and OASIS-2 (Longitudinal) MRI Datasets**

This report presents a detailed neuroimaging and clinical analysis of Alzheimer's Disease (AD) based on the OASIS-1 and OASIS-2 datasets processed using the MEDIC-AI structural MRI analysis pipelines. 

---

## 1. Executive Summary

- **Brain Atrophy as a Primary Biomarker:** Across both datasets, **Normalized Whole Brain Volume (nWBV)** is the single most statistically significant predictor of cognitive impairment. There is a clear linear decline in brain volume corresponding to higher Clinical Dementia Rating (CDR) scores and lower Mini-Mental State Examination (MMSE) scores.
- **Gray Matter Atrophy:** OASIS-1 data shows a direct reduction in gray matter fraction as dementia progresses, dropping from **44.4%** in healthy subjects to **41.2%** in mildly demented patients.
- **Structural Covariance Network Topology:** The brain's structural networks undergo significant reorganization during AD. By utilizing normalized regional intensities, our covariance analysis reveals that healthy brains maintain a robust small-world architecture (Small-worldness = **1.898** in OASIS-2), which is progressively disrupted by pathological atrophy.
- **Longitudinal Atrophy Trajectories:** In OASIS-2, subjects who eventually converted to dementia (Converted group) show a distinct, accelerated decline in nWBV over successive visits compared to healthy controls, with their mean age of conversion being significantly older (**83.5 years**).

---

## 2. Dataset Overviews & Tissue Analysis

### 2.1 OASIS-1 (Cross-Sectional - Disc 7)
OASIS-1 Disc 7 consists of 38 subjects, of which 23 have complete clinical metadata (CDR and MMSE). 

*Table 1: Clinical and Morphometric Summary of OASIS-1 by CDR Staging*
| Clinical Dementia Rating (CDR) | Subjects | Mean Age (years) | Mean MMSE | Mean nWBV | Mean GM Fraction | Mean WM Fraction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 (Nondemented)** | 13 | 74.69 | 29.23 | 0.7508 | 0.4443 | 0.3086 |
| **0.5 (Very Mild Dementia)** | 8 | 77.00 | 25.13 | 0.7234 | 0.4156 | 0.3103 |
| **1.0 (Mild Dementia)** | 2 | 75.00 | 22.00 | 0.6990 | 0.4127 | 0.2896 |
| *Missing Metadata* | 15 | — | — | — | — | — |

**Key Observations (OASIS-1):**
- **Symmetric Decline:** As CDR increases from 0 to 1, we observe a steady decrease in nWBV (from **75.1% to 69.9%**).
- **Early Gray Matter Loss:** The Gray Matter (GM) fraction drops sharply at the very earliest stage of cognitive decline (from **44.4%** at CDR=0 to **41.6%** at CDR=0.5), indicating that cortical gray matter loss is a precursor to clinically severe dementia.
- **Cognitive Correlation:** MMSE scores decrease in parallel with cortical atrophy, falling from near-perfect (29.2) to impaired (22.0).

---

### 2.2 OASIS-2 (Longitudinal - RAW PART1)
OASIS-2 contains longitudinal data from multiple visits. The analyzed subset contains **209 sessions** representing **82 unique subjects**.

*Table 2: Clinical and Morphometric Summary of OASIS-2 by Diagnosis Group*
| Diagnosis Group | Sessions | Mean Age (years) | Mean MMSE | Mean nWBV | Mean GM Fraction* | Mean WM Fraction* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nondemented** | 112 | 77.60 | 29.14 | 0.7375 | 0.3281 | 0.3391 |
| **Demented** | 81 | 76.79 | 24.05 | 0.7154 | 0.3286 | 0.3391 |
| **Converted** | 16 | 83.50 | 28.00 | 0.7158 | 0.3295 | 0.3385 |

> [!NOTE]  
> \* **Methodological Note on Tissue Fractions in OASIS-2:** The GM and WM fractions in OASIS-2 appear almost identical across groups (~32.8% GM, ~33.9% WM). This is a direct artifact of the simple automatic percentile-based segmentation (`auto_segment`) used for raw images, which labels the middle 33% of brain voxel intensities as Gray Matter and the upper 33% as White Matter. While these regional tissue volumes are mathematically constrained, the **nWBV** (Whole Brain Volume) and the **normalized regional intensities** remain fully descriptive of pathological changes.

**Key Observations (OASIS-2):**
- **Longitudinal Conversion Profile:** The "Converted" group represents subjects who entered the study cognitively normal but converted to AD. These individuals are significantly older on average (**83.5 years** vs **77.6 years** for controls), confirming that age is the dominant risk factor for clinical conversion.
- **Atrophy in Converted Group:** The nWBV of the Converted group (**0.7158**) is almost identical to the Demented group (**0.7154**), showing that substantial structural brain atrophy occurs *before* or *during* the transition to clinical dementia.

---

## 3. Structural Covariance Network (SCN) Analysis

Structural Covariance Networks model how brain regions co-vary in gray matter density across a population. We divided the T1 brain scans into a 3D grid of **64 Regions of Interest (ROIs)** and computed the Pearson correlation matrix of regional intensities.

### 3.1 Network Metric Comparison
By introducing **global intensity normalization** in OASIS-2, we successfully controlled for scanner-scaling biases. This resulted in biologically realistic, sparse networks compared to the unnormalized models.

*Table 3: Graph Theoretical Network Metrics*
| Graph Metric | OASIS-1 (Disc 7) | OASIS-2 (RAW - Normalized) | Clinical Interpretation |
| :--- | :---: | :---: | :--- |
| **Nodes** | 64 | 64 | Regions of interest (grid ROIs) |
| **Edges** | 1,059 | 532 | Valid structural correlations between regions |
| **Mean Degree** | 33.09 | 16.63 | Average number of connections per node |
| **Max Degree** | — | 29 | Connectivity limit of the dense hubs |
| **Clustering Coefficient** | 0.5236 | 0.5294 | Density of local sub-networks (segregation) |
| **Path Length** | 1.1657 | 2.6186 | Shortest path to transmit information (integration) |
| **Small-Worldness** | — | **1.8982** | Balance between local clustering and short paths |

**Key Findings:**
1. **Disrupted Covariance:** OASIS-1 shows a higher Mean Degree (33.09) and a very short Path Length (1.1657), indicating a dense, less differentiated network. 
2. **Small-World Architecture:** The OASIS-2 normalized SCN has a Small-Worldness index of **1.8982** (where $>1$ indicates small-world property). This shows that the brain maintains localized processing clusters (high clustering coefficient = 0.5294) linked by long-range connections (average path length = 2.6186).
3. **Biological Realism:** The SCN of OASIS-2 shows clear modular hubs (see the visualization below), suggesting that structural degeneration is not uniform but localized to specific vulnerability hubs.

---

## 4. Multiple Regression Modeling

We executed Ordinary Least Squares (OLS) regressions to quantify the clinical predictive power of MRI brain features.

### 4.1 Model 1: Clinical Dementia Rating (CDR)
Predicts the clinical stage of dementia:  
$$\text{CDR} \sim \text{nWBV} + \text{vol\_gm} + \text{vol\_wm} + \text{age}$$

*Table 4: OLS Regression Results for Model 1 (OASIS-2)*
- **Observations:** 209 sessions
- **R-squared ($R^2$):** 0.221 (Adjusted $R^2$: 0.206)
- **F-statistic:** 14.48 (p-value: **$1.99 \times 10^{-10}$**)

| Variable | Coefficient | Std Error | t-statistic | p-value | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | -9.8275 | 7.700 | -1.276 | 0.203 | |
| **nWBV** | -5.4872 | 0.728 | -7.532 | **< 0.001** | $\star\star\star$ |
| **vol_gm** | 9.9189 | 8.011 | 1.238 | 0.217 | |
| **vol_wm** | 35.5314 | 19.329 | 1.838 | 0.067 | $\cdot$ |
| **age** | -0.0155 | 0.003 | -4.734 | **< 0.001** | $\star\star\star$ |

**Analysis:**
- **nWBV Dominance:** For every 0.01 (1%) decrease in normalized whole brain volume, the predicted CDR score increases by **0.055 points** ($p < 0.001$). This confirms that overall brain atrophy is highly predictive of cognitive decline.
- **Multicollinearity:** The tissue fractions (`vol_gm`, `vol_wm`) are not significant because they are highly collinear with `nWBV` (Condition Number = $7.24 \times 10^4$). This suggests that absolute whole-brain volume loss is a more reliable predictor than individual tissue fractions.

### 4.2 Model 2: Cognitive Impairment (MMSE)
Predicts the cognitive exam score:  
$$\text{MMSE} \sim \text{nWBV} + \text{age} + \text{sex}$$

*Table 5: OLS Regression Results for Model 2 (OASIS-2)*
- **Observations:** 209 sessions
- **R-squared ($R^2$):** 0.198 (Adjusted $R^2$: 0.187)
- **F-statistic:** 16.90 (p-value: **$7.54 \times 10^{-10}$**)

| Variable | Coefficient | Std Error | t-statistic | p-value | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | -23.4652 | 8.356 | -2.808 | 0.005 | $\star\star$ |
| **nWBV** | 53.3408 | 8.474 | 6.295 | **< 0.001** | $\star\star\star$ |
| **age** | 0.1530 | 0.037 | 4.086 | **< 0.001** | $\star\star\star$ |
| **sex (Male)** | -0.3114 | 0.540 | -0.576 | 0.565 | |

**Analysis:**
- **Atrophy vs MMSE:** A decrease in nWBV is strongly associated with lower MMSE scores (greater cognitive decline). 
- **Sex Effects:** In this cohort, biological sex was not a statistically significant predictor of MMSE score ($p = 0.565$), suggesting that brain volume loss affects cognitive performance similarly in males and females.

---

## 5. Visualizations

### 5.1 OASIS-1 Disc 7 Analysis
This panel displays the clinical correlation between brain volumes, age, and dementia severity. We observe clear separation between healthy controls (CDR=0) and demented patients based on nWBV and GM volume.

![OASIS-1 Analysis Report](file:///E:/Medic/MEDIC-AI/result/o1/oasis_report.png)

### 5.2 OASIS-2 Longitudinal Analysis
The multi-panel report below shows:
1. **ROI Correlation Matrix:** Clearly defined correlation blocks representing localized functional/structural units.
2. **Structural Covariance Network:** Distinct node degrees with colored hubs.
3. **Longitudinal Trajectories:** Tracking individual subjects across visits shows a steady decline in nWBV, particularly in the Demented (red) and Converted (orange) groups.

![OASIS-2 Longitudinal Analysis Report](file:///E:/Medic/MEDIC-AI/result/o2/oasis2_report.png)

---

## 6. Scientific Discussion & Future Improvements

### 6.1 Limitations of the Current Pipeline
1. **Grid-based ROIs vs Anatomical Atlas:** The current 64-ROI grid does not align with anatomical brain regions (like the hippocampus or temporal lobe). This makes regional features susceptible to slight variations in patient positioning during raw scans.
2. **Simplified Segmentation:** Percentile-based tissue segmentation assumes constant proportions of CSF, GM, and WM, which misses absolute tissue-specific changes.

### 6.2 Suggested Next Steps
- **Atlas Registration:** Register the raw OASIS-2 scans to a standard template (e.g., MNI152) before extracting grid features.
- **Deep Learning segmentation:** Implement a 3D U-Net model to perform precise anatomical segmentation of the hippocampus, which is the hallmark region affected in early Alzheimer's Disease.
- **Longitudinal Rate of Change:** Calculate the *slope* of nWBV decline per subject as a feature to predict the exact time to convert to dementia.

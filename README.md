[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11479654.svg)](https://doi.org/10.5281/zenodo.11479654)

# hirsch_etal_2024_ef

**Two-Way Option Contracts that Enhance Adaptive Water Reallocation in the Western United States **

Zachary M. Hirsch<sup>1\*</sup>, Harrison B. Zeff <sup>1</sup>, Rohini S. Gupta <sup>2</sup>, Chris R. Vernon <sup>3</sup>, Patrick M. Reed <sup>2</sup>, and Gregory W. Characklis <sup>1</sup>

<sup>1 </sup> University of North Carolina at Chapel Hill, Center of Financial Risk in Environmental Systems
<sup>2 </sup> Cornell University, Department of Civil and Environmental Engineering
<sup>3 </sup> Pacific Northwest National Laboratory

\* corresponding author:  zacharyhirsch1@yahoo.com

## Abstract
Many water markets in the Western United States (U.S.) have the ability to reallocate water temporarily during drought, often as short-term water rights leases from lower value irrigated activities to higher value urban uses. Regulatory approval of water transfers, however, typically takes time and involves high transaction costs that arise from technical and legal analyses, discouraging short-term leasing. This leads municipalities to protect against drought-related shortfalls by purchasing large volumes of infrequently used permanent water rights. High transaction costs also result in municipal water rights rarely being leased back to irrigators in wet or normal years, reducing agricultural productivity. This research explores the development of a multi-year two-way option (TWO) contract that facilitates leasing from agricultural-to-urban users during drought and leasing from urban-to agricultural users during wet periods. The modeling framework developed to assess performance of the TWO contracts includes consideration of the hydrologic, engineered, and institutional systems governing the South Platte River Basin in Colorado where there is growing competition for water between municipalities (e.g., the city of Boulder) and irrigators. The modeling framework is built around StateMod, a network-based water allocation model used by state regulators to evaluate water rights allocations and potential rights transfers. Results suggest that the TWO contracts could allow municipalities to maintain supply reliability with significantly reduced rights holdings at lower cost, while increasing agricultural productivity in wet and normal years. Additionally, the TWO contracts provide irrigators with additional revenues via net payments of option fees from municipalities.

## Journal reference

Hirsch, Z. M., Zeff, H. B., Gupta, R. S., Vernon, C. R., Reed, P. M., & Characklis, G. W. “Two-Way Option Contracts that Facilitate Adaptive Water Reallocation in the Western United States,” Earth’s Future (in review)

## Code reference

Hirsch, Z.M. (2024). IMMM-SFA/hirsch_etal_2024_ef: v1.0.0. Zenodo. https://doi.org/10.5281/zenodo.11479654.

## Contributing modeling software
| Model | Version | Link |
|-------|---------|-----------------|
| StateMod | v15 | https://cdss.colorado.gov/software/statemod |

## Reproduce my experiment*

Organize three separate folders on your local machine. 
1) for the UCRB StateMod model (for pre-processing)
2) for the SPRB StateMod model (for pre-processing)
3) for the adapted SPRB StateMod model (for post-processing)

*All working directories must be changed in order for the code to run.

## Pre-processing + StateMod
1. Download the StateMod Model 15.00.01 Executable from: https://cdss.colorado.gov/software/statemod
2. Download the South Platte and Upper Colorado StateMod input files from: https://cdss.colorado.gov/modeling-data/surface-water-statemod
3. Run StateMod v15 for both the Upper Colorado and South Platte (using simulate, option 2)
4. If South Platte StateMod fails to run, try replacing the .opr file to 'SP2016_rev1501.opr' which can be downloaded here: https://github.com/OpenCDSS/cdss-app-statemod-fortran/issues/73
5. Call 'statemod_data_extraction.py' to extract StateMod demand data (.xdd) into parquet files. Change the name of the .xdd file to 'sp2016_H_S0_1.xdd'. You must use an 'ids' file to indicate which StateMod structures you want a parquet for. Sample code to call this: python statemod_data_extraction.py --ids C:\path\ids_file.txt --output C:\path\xddparquet C:/path/sp2016_H_S0_1.xdd
5. Use the scripts and tools in the 'StatemodAdaptations&Tools' folder to update StateMod input files
6. Re-run the South Platte StateMod to produce new outputs for use in post-processing
7. *All resulting parquet files from StateMod adaptations are included in the 'xddparquet' and 'xreparquet' folder

## Post-processing
1. Run one of the following scripts to see results from a given rights regime and pricing scenario (housed in 'FinalTWOScripts' with all other scripts called in a regime/scenario script)

| Script Name | Description |
| --- | --- |
| `TWO_update_S1_manuscript_final.py` | Script to run the two-way option through the current rights regime, pricing scenario 1 |
| `TWO_historical_S1_manuscript_final.py` | Script to run the two-way option through the historical (1971) rights regime, pricing scenario 1 |
| `TWO_update_S2_manuscript_final.py` | Script to run the two-way option through the current rights regime, pricing scenario 2 |
| `TWO_historical_S2_manuscript_final.py` | Script to run the two-way option through the historical (1971) rights regime, pricing scenario 2 |
| `TWO_update_S3_manuscript_final.py` | Script to run the two-way option through the current rights regime, pricing scenario 3 |
| `TWO_historical_S3_manuscript_final.py` | Script to run the two-way option through the historical (1971) rights regime, pricing scenario 3 |
| `TWO_update_S4_manuscript_final.py` | Script to run the two-way option through the current rights regime, pricing scenario 4 |
| `TWO_historical_S4_manuscript_final.py` | Script to run the two-way option through the historical (1971) rights regime, pricing scenario 4 |

## Reproduce my figures
Use the scripts found throughout this analysis to reproduce the figures used in this publication. Final figures are provided in the 'figures' folder.

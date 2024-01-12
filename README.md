_your zenodo badge here_

# hirsch_etal_2023_ef

**Two-Way Option Contracts that Enhance Adaptive Water Reallocation in the Western United States **

Zachary M. Hirsch<sup>1\*</sup>, Harrison B. Zeff <sup>1</sup>, Rohini S. Gupta <sup>2</sup>, Chris R. Vernon <sup>3</sup>, Patrick M. Reed <sup>2</sup>, and Gregory W. Characklis <sup>1</sup>

<sup>1 </sup> University of North Carolina at Chapel Hill, Center of Financial Risk in Environmental Systems
<sup>2 </sup> Cornell University, Department of Civil and Environmental Engineering
<sup>3 </sup> Pacific Northwest National Laboratory

\* corresponding author:  zacharyhirsch1@yahoo.com

## Abstract
Many water markets in the Western United States (U.S.) have the ability to reallocate water temporarily during drought, often as short-term water rights leases from lower value irrigated activities to higher value urban uses. Regulatory approval of water transfers, however, typically takes time and involves high transaction costs that arise from technical and legal analyses, discouraging short-term leasing.  This leads municipalities to protect against drought-related shortfalls by purchasing large volumes of infrequently used permanent water rights. High transaction costs also result in municipal water rights rarely being leased back to irrigators in wet or normal years, reducing agricultural productivity. This research explores the development of a multi-year two-way option (TWO) contract that facilitates leasing from agricultural-to-urban users during drought and leasing from urban-to agricultural users during wet periods. The modeling framework developed to assess performance of the TWO contracts includes consideration of the hydrologic, engineered, and institutional systems governing the South Platte River Basin in Colorado where there is growing competition for water between municipalities (e.g., the city of Boulder) and irrigators. The modeling framework is built around StateMod, a network-based water allocation model used by state regulators to evaluate water rights allocations and potential rights transfers.  Results suggest that the TWO contracts could allow municipalities to maintain supply reliability with significantly reduced rights holdings at lower cost, while increasing agricultural productivity in wet and normal years. Additionally, the TWO contracts provide irrigators with additional revenues via net payments of option fees from municipalities.  

## Journal reference
Edmonds, J., & Reilly, J. (1983). A long-term global energy-economic model of carbon dioxide release from fossil fuel use. Energy Economics, 5(2), 74-88. DOI: https://doi.org/10.1016/0140-9883(83)90014-2

## Code reference
References for each minted software release for all code involved.  

These are generated by Zenodo automatically when conducting a release when Zenodo has been linked to your GitHub repository. The Zenodo references are built by setting the author order in order of contribution to the code using the author's GitHub user name.  This citation can, and likely should, be edited without altering the DOI.

If you have modified a codebase that is outside of a formal release, and the modifications are not planned on being merged back into a version, fork the parent repository and add a `.<shortname>` to the version number of the parent and construct your own name.  For example, `v1.2.5.hydro`.

Human, I.M. (2021, April 14). Project/repo:v0.1.0 (Version v0.1.0). Zenodo. http://doi.org/some-doi-number/zenodo.7777777

## Data reference

### Input data
Reference for each minted data source for your input data.  For example:

Human, I.M. (2021). My input dataset name [Data set]. DataHub. https://doi.org/some-doi-number

### Output data
Reference for each minted data source for your output data.  For example:

Human, I.M. (2021). My output dataset name [Data set]. DataHub. https://doi.org/some-doi-number

## Contributing modeling software
| Model | Version | Repository Link | DOI |
|-------|---------|-----------------|-----|
| StateMod | v15 | link to code repository | link to DOI dataset |
| model 2 | version | link to code repository | link to DOI dataset |
| component 1 | version | link to code repository | link to DOI dataset |

## Reproduce my experiment

## Pre-processing
1. Download the StateMod Model 15.00.01 Executable from: https://cdss.colorado.gov/software/statemod
2. Download the South Platte and Upper Colorado StateMod input files from: https://cdss.colorado.gov/modeling-data/surface-water-statemod
3. Run the following scripts in the 'pre_processing' file to update StateMod input files
4. Re-run the South Platte StateMod to produce new outputs for use in post-processing

| Script Name | Description | How to Run |
| --- | --- | --- |
| `update_rsp.py` | Script to run the first part of my experiment | `python3 step_one.py -f /path/to/inputdata/file_one.csv` |
| `ddm_extraction_timeseries.py` | Script to run the second part of my experiment | `python3 step_two.py -o /path/to/my/outputdir` |

4. Download and unzip the output data from my experiment [Output data](#output-data)
5. Run the following scripts in the `workflow` directory to compare my outputs to those from the publication

| Script Name | Description | How to Run |
| --- | --- | --- |
| `compare.py` | Script to compare my outputs to the original | `python3 compare.py --orig /path/to/original/data.csv --new /path/to/new/data.csv` |

## Reproduce my figures
Use the scripts found in the `figures` directory to reproduce the figures used in this publication.

| Script Name | Description | How to Run |
| --- | --- | --- |
| `generate_figures.py` | Script to generate my figures | `python3 generate_figures.py -i /path/to/inputs -o /path/to/outuptdir` |

# The PETALE Project
This repository stores the code implemented to generate the results of our paper:  
*Machine learning strategies to predict late adverse effects in childhood acute lymphoblastic leukemia survivors*

**The datasets analysed during the study are not publicly available for confidentiality purposes.
However, randomly generated datasets with the same format as used in our experiments are publicly
shared in the** ```data``` **directory**.

## Installation
To have all the requirements needed, you must do the following actions:
- Open a terminal
- Clone this repo: ```git clone https://github.com/Rayn2402/ThePetaleProject.git```
- Move into the directory: ```cd ThePetaleProject/```
- Create a virtual environment with conda: ```conda env create --file settings/env.yml```
- Activate your new environment: ```conda activate petale```
  
## Test the implementation
You can write the following lines in a terminal to replicate our experiments using the **randomly generated** data stored
in the ```data``` directory. Records of the experiments will be stored in ```records/experiments``` directory
as they will be completed.

- VO2 peak:  
```python replicate_study.py -d generated -t vo2```
  
- Obesity (w/o SNPs):  
```python replicate_study.py -d generated -t obesity```
  
- Obesity (w/ SNPs):  
```python replicate_study.py -d generated -t obesity -gen```
  
It is also possible to run a fast version of each experiment by adding the ```-fast``` argument.
The fast version uses **2** rounds of stratified random sampling, instead of **10**,
to evaluate each model and hyperparameter sets.

Specs of our computer and execution times recorded for each experiment
are displayed below.   

- Computer model:  Alienware Aurora Ryzen Edition
- Linux version: Ubuntu 20.04.4 LTS
- CPU: AMD Ryzen 9 3900X 12-Core Processor
- GPU: None were used for our experiments

| Experiment         | Time (fast)    | Time (normal)    |
|--------------------|----------------|------------------|
| VO2                | 42 min         | 4 hours 11 min.  |
| Obesity (w/o SNPs) | 1 hour 1 min.  | 7 hours 8 min.   |
| Obesity (w/ SNPs)  | 2 hours 7 min. | 16 hours 39 min. |

## What's new in this fork?

In this version of the repository, the scripts associated to the creation of data tables
specific to the projet are available in the ```scripts/tables_creation directory```. 

Additionally, a script created to upload all PETALE data tables on postgreSQL is available
in the ```scripts/petale_importation directory```. To be executed, this script
requires to do the following actions :

1. Download the PETALE folder containing all the original .xlsx files;
2. Create an empty postgreSQL database that will store the data tables;
3. Place the script within the folder you downloaded in step #1;
4. Run the script from a terminal (see the script for the possible arguments).

```diff
- WARNING
- If an error occurs during the execution of the importation script,
- incorrect characters in the original .xlsx files might be in cause.
```
```diff
! NOTE
! The importation script was created before the repository of the PETALE project itself.
! Hence, the code within the file petale_tables_importation_script.py does not follow
! the actual coding standards of MEDomics. Moreover, the DataManagement module provided 
! in the file src/data/extraction/data_management.py could potentially be used to 
! replace some parts of this script. An update of the importation script should be done.
```


## Project Tree
```
├── checkpoints                   <- Temporary state dictionaries save by the EarlyStopper module
├── data
│   ├── obesity_dataset.csv       <- Synthetic dataset for the obesity prediction task
│   └── vo2_dataset.csv           <- Synthetic dataset for the VO2 peak prediction task
|
├── hps                           <- Python files used to store sets of hyperparameter values and search spaces
├── masks                         <- JSON files used to store random stratified sampling masks
├── models                        <- State dictionaries associated to the best models
├── records                       <- Directories in which results and summaries of data analyses are stored
|
├── scripts
│   ├── experiments               <- Scripts to run individual experiments
|   ├── petale_importation        <- Script used to upload PETALE tables in PostgreSQL
│   ├── post_analyses             <- Scripts to run post analyses
|   ├── tables_creation           <- Scripts used to create new tables in the PETALE database.
│   └── utils                     <- Scripts to execute different sub tasks
|
├── settings                      <- Files used for the setup of the project environment
|
├── src                           <- All project modules
│   ├── data
│   │   ├── extraction            <- Modules related to data extraction from PostgreSQL
│   │   └── processing            <- Modules related to data processing
│   ├── evaluation                <- Modules related to the evaluation and tuning of the models
│   ├── models
│   │   ├── abstract_models       <- Abstract classes from which new models have to inherit
│   │   ├── blocks                <- Neural network architecture blocks
│   │   └── wrappers              <- Abstract classes used to wrap existing models
│   ├── recording                 <- Recording module
│   └── utils                     <- Modules associated to visualization, metrics, hps and more
├── replicate_study.py            <- Main script used to replicate the experiments of the study
└── README.md
```

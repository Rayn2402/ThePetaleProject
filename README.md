# ThePetaleProject
## Project Tree
```bash
.
├── Datasets
│   ├── Datasets.py                     <- Custom Datasets classes
│   ├── Preprocessing.py                <- Preprocessing functions
│   ├── Sampling.py                     <- Sampler classes
│   └── Transforms.py                   <- Custom transforms for preprocessing
│
├── DescriptiveAnalyses
│   └── DescriptiveTablesScript.py      <- Script to get descriptive analyses out of SQL tables
│
├── Evaluator
│   └── Evaluator.py                    <- Class that manages nested cross validation
│
├── Experiments
│   └── WarmUp
│       ├── LinearRegressionExp.py      <- Linear Regression experiment on WarmUp dataset
│       ├── NNExp.py                    <- NNRegressor experiment on WarmUp dataset
│       └── Results                     <- Results of WarmUp experiments
│
├── Hyperparameters
│   ├── RF_hyper_params.json            <- Hyperparameters of Random Forest
│   ├── constants.py                    <- Constants used for hyperparameter optimization
│   └── hyper_params.json               <- Hyperparameters of Neural Nets
│
├── Models
│   ├── GeneralModels.py                <- Models classes for Neural Nets
│   ├── LinearModel.py                  <- Linear regression models
│   └── ModelGenerator.py               <- Class that generates model during hyperparameter optimization
│
├── Recorder
│   └── Recorder.py                     <- Class that records experiments insights
│
├── SQL
│   ├── DataManager
│   │   ├── ChartServices.py            <- Vizualisation functions for DataManager
│   │   ├── Helpers.py                  <- Helpers function for DataManager
│   │   └── Utils.py                    <- Store the DataManager class managing interaction with SQL
│   │  
│   └── NewTablesScripts                <- Scripts that generates table for our experiments
│   │   ├── 6MWT.py
│   │   ├── DEX_DOX.py
│   │   ├── GENERALS.py
│   │   ├── L0_WARMUP.py
│   │   ├── L1_6MWT.py
│   │   ├── VO2_ID.py
│   │   ├── constants.py
│   │   └── csv_files
│   │       ├── DEX_DOX.csv
│   │       └── VO2_ID.csv
│
├── SanityChecks
│   └── NNClassifier
│       ├── HyperOpt.py             <- Hyperparameter optimization test with NNClassifier
│       └── Train.py                <- Training sanity checks with NNClassifier
│
├── Training
│   ├── EarlyStopping.py            <- Class that manages early stopping
│   ├── Training.py                 <- Classes that manage training
│
├── Tuner
│   └── Tuner.py                    <- Tuner class that deals with hyperparameter optimization
│
├── Utils
│   ├── score_metrics.py            <- Metrics to evaluate models
│   └── visualization.py            <- Visualization functions

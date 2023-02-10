# structural_health_monitoring (WORK IN PROGRESS)

[Project](https://zenodo.org/record/) **|** [Paper](https://)


[Christoph Kralovec](https://www.jku.at/en/institute-of-structural-lightweight-design/team/christoph-kralovec/),
[Bernhard Lehner](https://www.researchgate.net/profile/Bernhard_Lehner),
[Martin Schagerl](https://www.jku.at/en/institute-of-structural-lightweight-design/team/martin-schagerl/),

Sandwich Face Layer Debonding Detection and Size Estimation
by Machine Learning-based evaluation of Electromechanical
Impedance Measurements, Sensors, 2023

## Abstract
The present research proposes a two-step physics- and machine learning (ML)-based electromechanical impedance (EMI) measurement data evaluation approach for sandwich face layer debonding detection and size estimation in structural health monitoring applications.
As a case example, a circular aluminum sandwich panel with idealized face layer debonding is used.
Both, sensor and debonding are located at the center of the sandwich.
Synthetic EMI spectra are generated by a finite element (FE)-based parameter study and used for feature engineering and ML model training and development.
Calibration of the real-world EMI measurement data is shown to overcome the FE-model simplifications, enabling their evaluation by the found synthetic data-based features and models.
Data preprocessing and ML models are validated by unseen real-world EMI measurement data collected in laboratory environment.
The best detection and size estimation performances were found for a One-class Support Vector Machine and a K-Nearest Neighbor model, respectively, which clearly show reliable identification of relevant debonding sizes.
Furthermore, the approach is shown to be robust against unknown artificial disturbances and outperforms a previous method for debonding size estimation.
The data and code used in this study are provided in their entirety to enhance comprehensibility and encourage future research.


## Table of Contents
1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Reproducing Results](#reproduction)


## Introduction <a name="introduction"></a>
TODO

## Reproducing Results <a name="reproduction"></a>
The notebooks contain a Colab badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)], which can be clicked to open a Colab runtime to reproduce the results without the need to locally install anything. More information about Colab can be found here: [www.tutorialspoint.com/google_colab](https://www.tutorialspoint.com/google_colab/index.htm).


1.1. [Feature Engineering: Data Exploration](https://github.com/berni-lehner/structural_health_monitoring/blob/main/notebooks/data_exploration.ipynb)

1.2. [Feature Engineering: Filterbank Design](https://github.com/berni-lehner/structural_health_monitoring/blob/main/notebooks/feature_engineering.ipynb)

1.3. [Feature Engineering: Discrete Cosine Transform](https://github.com/berni-lehner/structural_health_monitoring/blob/main/notebooks/feature_engineering_2.ipynb)

1.4. [Calibration](https://github.com/berni-lehner/structural_health_monitoring/blob/main/notebooks/data_calibration.ipynb)

2.1. [Anomaly Detection: Setting up Cross-Validation](https://github.com/berni-lehner/structural_health_monitoring/blob/main/notebooks/synthetic_anomaly_AA.ipynb)

2.2. [Anomaly Detection Experiment: Synthetic Data](https://github.com/berni-lehner/structural_health_monitoring/blob/main/notebooks/synthetic_anomaly_AB.ipynb)

2.2.1. [Anomaly Detection Results: Synthetic Data](https://github.com/berni-lehner/structural_health_monitoring/blob/main/notebooks/synthetic_anomaly_results.ipynb)

2.3. [Anomaly Detection Experiment: Real-World Data](https://github.com/berni-lehner/structural_health_monitoring/blob/main/notebooks/mixed_anomaly_AB.ipynb)

2.3.1. [Anomaly Detection Results: Real-World Data (TODO)]

3.1. [Damage Size Estimation: Setting up Cross-Validation (TODO)]

3.2. [Damage Size Estimation: Synthetic Data (TODO)]

3.3. [Damage Size Estimation: Real-World Data (TODO)]




## Citation <a name="citation"></a>
If you find the code and datasets useful in your research, please cite:
    
    @article{kralovec2023sen,
         title={Sandwich Face Layer Debonding Detection and Size Estimation by Machine Learning-based evaluation of Electromechanical Impedance Measurements},
         author={Kralovec, C. and Lehner, B. and Kirchmayr, M. and Schagerl, M.},
         journal={Sensors},
         pages={--},
         year={2023}
         volume={1},
         number={1},
         year={2023},
         publisher={MDPI}
    }    

# BP Estimation Slapnicar

## Project Description
The present repository implements the data sorting, preprocessing, and training from the work of Slapnicar et al [1]. It describes a blood pressure estimation from PPG data using the MIMIC 3 dataset and various machine learning methods.

## About The Project
The project consists of classes that enable the overall data processing. In addition, in the folder 'demos', there are demonstrations of data processing as they were performed in the project. The 'notebooks' folder contains minimal examples of the use of the repository.

## Built With
- numpy
- sklearn
- tensorflow
- scipy
- os-sys
- wfdb

## Getting Started
1. Clone the repository.
2. Change paths which are indicated in the scripts to match your environment.
3. Create a new folder structure for newly generated data.

## Usage
The following scripts of the demos must be executed in the specified order:
1. `demo_first_segmentation.py`
2. `demo_preprocessing.py`
3. `demo_extract_cycles.py`
4. `demo_extract_feat_ground_truth.py`
5. `demo_make_data_nn.py`

The remaining demos contain the methods of machine learning and are independent of any sequence.

## Acknowledgments
[1] Slapnicar, G., Mlakar, N. & Lustrek, M. (2019) Blood Pressure Estimation from Photopletysmogram Using a Spectro-Temporal Deep Neural Network. MDPI(19): 3420. doi:10.3390/s19153420.

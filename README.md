# Hyperparameter_optimization

In this project three algorithms were compared to optimize the hyperparameters of a neural network.

*More details can be found in the pdf in this repository*

## Installation

1. Clone this repository with:
```sh
git clone https://github.com/ScaramuzzinoGiovanna/Hyperparameter_optimization.git
```
2. Prerequisites for run this code:
    * OpenCV  
    * Keras   
    * numpy 
    * scikit-learn
    * argparse 
    * imutils
    * matplotlib
    * bayesian-optimization
    * rbfopt 
    
   You can install these using pip or Anaconda

## Usage for the user

* Run the code insert to command line, inside the directory containing the files:
    ```sh
    python main.py -t <type of optimizator> ( insert: rbf, bayesian or random - default value is rbf)
    ```
  **Note**: if you are not familiar with the command line can you run the file *main.py* and manually edit the optimizer by changing                 the default parameter in the method ap.add_argument

  A part of the [Breast-histopatology](https://www.kaggle.com/paultimothymooney/breast-histopathology-images) dataset was                 used, inserted in the 'db_60000' folder, dividend into train (48000 images) and test (12000 images). These was created                   by 'generate_db.py'<br>

* Optional: 
    - Download the [Breast-histopatology](https://www.kaggle.com/paultimothymooney/breast-histopathology-images)  dataset <br>    
    - Insert the Breast-histopathology dataset, downloaded previously, into a 'db' folder<br> 
    - To create a new dataset, subset of Breast-histopathology: run 'generate_db.py' by command line and set the parameter <*sizeDB*>.         This will create a reduced dataset and divide it into train and test. Then remember to change the name of the dataset to be used         in main.py
  
## Requirements
The use of a GPU is recommended

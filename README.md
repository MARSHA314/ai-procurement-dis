# AI Utilisation for material procurement

## Description
This is a repository for a code that was used to analyse data for dissertation on topic of AI implementation for material procurement. It is split into two parts: EDA - data transformations and analysis and ML - code used to train neural networks machine learning model and analyse results. Sample of anonymized data is provided in the data folder.

The aim is to utilise neural networks to analyse, given indicators in the data, if contract could be purchased automatically at standard quantity (quantity set up by procurement process, dependent on type of material and vendor) or if buyers' intervention was needed. As such the problem is classification problem with classes Standard, Low (when drop of quantity is needed to avoid excess) and High (when higher quantities could be purchased to improve price per piece).

Originally the data were trained at dataset containing materials aimed for production of finish good. These materials have 1:1 relationship with the finish good, ie box packaging with printing of the finish good on it. The size of the original dataset is 23000x49. Current results are that the data are not strongly correlated, therefore sensitive feature selection needed to be introduced to remove noise, and that the data are highly imbalanced toward Standard class. Standard class constitutes 60% of the dataset. SMOTE was introduced to deal with the imbalanced classes.

Currently the accuracy of the model is 67%, but initial overfitting towards standard class was addressed by inclusion of SMOTE and feature selection. The final preprocessor and model are saved in model folder.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation
1. Clone the repository:
    ```bash
    https://github.com/MARSHA314/ai-procurement-dis/
    ```
2. Navigate to the project directory:
    ```bash
    cd your-repo-name
    ```
3. Install dependencies:
   #Python version: 3.11.9 
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the project, run the python files in order:

1.  Anonymise w dict.py
    This code cleans the data and anonymises them, while creating dictionary that is saved to model/mappings.pkl.
2.  EDA_stat_cond_2.py
    This code creates variable contract weight and saves descriptive statistics for the data set that are saved in graphs folder. Condition to filter outliers is included. 
3.  EDA_dec_var_3.py
    This program creates graphs and descriptive statistics for analysis of the decision variable and saves them in graphs folder.
4.  EDA_category_4.py
    This program creates graphs for analysis of relationship between decision variable and categorical data and saves them in graphs folder.
5.  EDA_correlation_5.py
    This program creates graphs for analysis of relationship between decision variable and numerical data and saves them in graphs folder.
6.  ML_train_strat_6.py
    This program trains the neural network with all data. Outputs results, confusion matrix and loss function. Model is saved in model folder marked with '_plain'. 
7.  ML_train_strat_k_fold_7
    This program trains the neural network with all data utilising the k-fold method, where k=10. Outputs results.
8.  ML_train_strat_w_feat_sel_8.py
    This program trains the neural network and selects top 75% to reduce noise. Outputs results, confusion matrix and loss function. Model is saved in model folder marked with      '_feat_sel'. 
9.  ML_train_strat_k_fold_9.py
    This program trains the neural network with feature selection utilising the k-fold method, where k=10. Outputs results.
10. ML_train_strat_w_feat_sel_class_imb_10.py
    This program trains the neural network and selects top 75% to reduce noise, decision variable classes are balanced using SMOTE. Outputs results, confusion matrix and loss function. Model is saved in model folder marked with '_feat_sel_smote'. 
11. ML_train_strat_w_feat_sel_class_imb_k_fold_11.py
    This program trains the neural network with feature selection and SMOTE utilising the k-fold method, where k=10. Outputs results.



## Contributing
Explain how others can contribute to your project. Include guidelines for submitting issues and pull requests.

## License


## Contact
Created by [Marie Shazad](https://yourwebsite.com) - feel free to contact me!

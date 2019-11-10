Blue Book for Bulldozers

The objective was to Predict the auction sale price for a piece of heavy equipment to create a "blue book" for bulldozers.
The datasets used was split into two parts:

TrainAndValid.csv: This dataset was used for the training of the model. This dataset contained the data through the end of 2011 and from January 1, 2012 - April 30, 2012.
Test.csv: This dataset was used for the final prediction of the final model. It contains data from May 1, 2012 - November 2012.

Key data fields in the training dataset are as follows:

SalesID: The Unique Identifier of the sale.
MachineID: The unique identifier of a machine. Note: A machine can be sold multiple times.
saleprice: The price at which the machine was old for at auction (only provided in training dataset and also our target variable).
saledate: The date of the sale.
METRIC:
The model was evaluated on the Root Mean Square Logarithmic Error (RMSLE) between the predicted value and observed score values.
This folder contains model, jupyter notebook file (training code) and python script which predicts classes when run in terminal with location of the test file as argument.

Please use make_conda_env.sh bash script for making conda environment with specific dependancies.

Example on how to get predictions:

$ python bulldozer.py 

Output - CSV file containing predictions.

Note: The train and test files should be in the same repository and the final predictions made on the test dataset was not submitted onto Kaggle for evaluation due to the unavailability of the Submission Tab.
make_conda_env_sh.sh - Bash Script for making conda environment and installing all dependancies.

The dataset has been obtained from a Kaggle competition and can be found by clicking on the link of README.md .

The RMSLE score obtained for the Baseline model was 0.2381716487809174
The RMSLE score obtained obtained after tuning the hyperparameters was: 0.24881865963344546
From this it has been noticed that the results have minimal improvement after tuning the hyperparameters.

Future Improvements:
Further fine tuning can be done by selecting a differnt combinations of features and perhaps trying other ensembling techniques such as Stacking.

The main problems faced was with the data itself. There was a high percentage of missing values in 53 features of the dataset. The values were either missing or were erroneous.

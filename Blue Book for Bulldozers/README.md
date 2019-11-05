## Blue Book for Bulldozers

![Snapshot](https://user-images.githubusercontent.com/36225987/68218369-365da080-000a-11ea-8778-80bf14d23b5a.JPG)


The objective was to Predict the auction sale price for a piece of heavy equipment to create a "blue book" for bulldozers. <br>
The datasets used was split into two parts:
<ul>
<li><b>TrainAndValid.csv</b>: This dataset was used for the training of the model. This dataset contained the data through the end
of 2011 and from January 1, 2012 - April 30, 2012</li>
<li><b>Test.csv</b>: This dataset was used for the final prediction of the final model. It contains 
data from May 1, 2012 - November 2012.</li>
</ul>
The dataset has been obtained from a Kaggle competition and can be found by clicking
<a href="https://www.kaggle.com/c/bluebook-for-bulldozers/data">here</a>.</font><br>
<h2> Key data fields in the training dataset are as follows:</h2>
<ul>
<li><b>SalesID: </b>The Unique Identifier of the sale.</li>
<li><b>MachineID: </b>The unique identifier of a machine. Note: A machine can be sold multiple times.</li>
<li><b>saleprice: </b>The price at which the machine was old for at auction (only provided in training dataset and also our target variable).</li>
<li><b>saledate: </b>The date of the sale.</li></ul>
<h2> METRIC:</h2>
The model was evaluated on the Root Mean Square Logarithmic Error (RMSLE) between the predicted value and observed score values. 
<br><br>

### This folder contains model, jupyter notebook file (training code) and python script which predicts classes when run in terminal with location of the test file as argument.

Please use make_conda_env.sh bash script for making conda environment with specific dependancies.

Example on how to get predictions:
```
$ python bulldozer.py 
```
Output - CSV file containing predictions.
<br>
<b>Note: </b>The train and test files should be in the same repository and the final predictions made on the test dataset was not submitted onto Kaggle for evaluation due to the unavailability of the Submission Tab.<br>
<b>make_conda_env_sh.sh</b> - Bash Script for making conda environment and installing all dependancies.

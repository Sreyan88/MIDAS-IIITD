### This folder contains model, jupyter notebook file (training code) and python script which predicts classes when run in terminal with location of the test file as argument.

Please use make_conda_env.sh bash script for making conda environment with specific dependancies.

Example on how to get predictions:
```
$ python predict.py --input meld/test
```

### Note: The entire code and model was trained on PyTorch GPU. So cuda is necessary for running any of the files. Please ensure you are running this script on a system with GPU support.

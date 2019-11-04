%%writefile make_conda_env.sh

read -p "Create new conda env (y/n)?" CONT

if [ "$CONT" == "n" ]; then
  echo "exit";
else
# user chooses to create conda env
# prompt user for conda env name
  echo "Creating new conda environment, choose name"
  # read input_variable
  # echo "Name $input_variable was chosen";

  # Create environment.yml or not
  read -p "Create 'enviroment.yml', will overwrite if exist (y/n)?"
    if [ "$CONT" == "y" ]; then
      conda create --name test
      python=3 jupyter notebook numpy rpy2 pip\
      pandas scipy numpy scikit-learn seaborn\
      tqdm tensorflow=1.15 keras regex;
      conda install -c jmcmurray os;
      conda install -c conda-forge librosa;
      conda install -c peterjc123 pytorch-gpu;
      conda install -c conda-forge glob2;
      conda install -c contango python_speech_features

    else
        echo "installing base packages"
        conda create --name test\
        python=3.6 jupyter notebook numpy rpy2\
        pandas scipy numpy scikit-learn seaborn pip
    fi
  echo "to exit: conda deactivate $input_variable"
fi
# conda activate $input_variable
# pip install fastai==0.7.0
# pip install sklearn_pandas
# virtualenv -q -p /usr/bin/python3.5 $1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
pip install fastai==0.7.0
pip install sklearn_pandas

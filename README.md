# Customer Churn Prediction with Machine Learning

[Demo](https://youtu.be/Mz1d4B9IB4Y)

Description:


Skills: Python, OpenAI, Machine Learning, Model Inference, Feature Engineering, Data Visualization

## Part 1:
* Downloading the dataset
* Understanding the dataset
* Pre-processing the dataset
* Training the ML models
* Evaluating the ML models


## Part 2:
* Building the web app
    * Creating the UI and charts
    * Making predictions with the ML models
    * Generating personalized emails with OpenAI or LLama 3.1 via Groq


    Troubleshooting:
    * Update on storing serialized model and memory snapshots using Pickle
    Use `save_model` instead of `pickle.dump` to save the model
    https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html


![website1](public/assets/website1.png)
![website2](public/assets/website2.png)



# Developer's Notes
## Add the git repo:
```bash
git init
git add *
git commit -m 'Initial commit'
git branch -M main
git remote add orgin https://github.com/itancio/<repot>.git
git push -u origin main
```

## Create Python virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

## Running streamlit
```bash
cd streamlit
streamlit run main.py
```

# Troubleshooting:
* ERROR: The error was related to XGBoost failing to load due to the missing OpenMP runtime library (libomp.dylib) on macOS

SOLUTION: Update ```libomp```
```bash
brew install libomp
```

* ERROR: Handling large files
```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes

git add notebook/fraud/fraudTest.csv
git add notebook/fraud/fraudTrain.csv
git commit -m "Add files"
git push origin HEAD:fraud
```
**Prerequisites:**

*For running on GPU:*
11.2 cuda toolkit, cudnn 8.1.0, tensorflow-gpu 2.10.0 

*MLflow:*
Modeltraining runs on port:5000, Modelserving runs on port:5002

NOTE:
By default the script does not perform training. The expectations on the raw data, must return True in order for the training to occur. If you want to train a model, simply change the value_pair_set to [(2708,3384)] in check_expectations().

We have not trained a model for more than 1 epoch. It is just for testing if we can deploy and run inference on the deployed model. 

Download the model from GoogleDrive. The model can be donwlaoded from this link: https://drive.google.com/drive/folders/1obTbbVGhHvAKYvgBKI_t33XO3z5mTYKb?usp=sharing

download the folder into the src folder in the project. 

*Overview of main.py*
  1. Loads raw data from "raw_data/input", "raw_data/output"
  2. Uses great_expectations to check data quality
  3. Preprocesses the data in patches 256x256 (must patch the images to be able to fit data into VRAM)
  4. Versions patches to DVC 
  5. Trains a Unet model (30.000.000+ params), with MLflow
  6. Checks which model is better (both are not good, however it is for exemplifying a check on the performance before deploying)
  7. Deploys the best model
  8. Predicts on the model in ddeployment
  9. Saves some predictions from the deployed model





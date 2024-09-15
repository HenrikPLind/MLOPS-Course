import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
import time
from matplotlib.image import imread
from numpy.random import randint
from sklearn.utils import compute_sample_weight
from tensorflow.python.keras.metrics import MeanIoU
from tensorflow.python.keras.utils.np_utils import to_categorical, normalize
import os
import glob
from matplotlib import pyplot as plt, pyplot
import numpy as np
import cv2
import tensorflow.keras.backend as K
import tensorflow as tf
from models.models import multi_unet_model # uses softmax
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

EPOCHS = 100
BATCH_SIZE = 16
earlyStoppingVar = 1000
n_classes = 3

def create_sample_weights(label_mask_path):
    output_train = []
    for directory_path in glob.glob(label_mask_path):
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            img = cv2.imread(img_path, 0)  # the 0 means we read as grayscale!
            output_train.append(img)

            # Data augmentation - flipping horizontally and vertically
            mask_flipped_horiz = cv2.flip(img, 0)
            output_train.append(mask_flipped_horiz)
            mask_flipped_vert = cv2.flip(img, 1)
            output_train.append(mask_flipped_vert)
            mask_flipped_vert_horiz = cv2.flip(mask_flipped_vert, 0)
            output_train.append(mask_flipped_vert_horiz)

            output_train = np.array(output_train)

            n, h, w, y = output_train.shape
            list = output_train.reshape(-1, 1)  # y in compute_sample_weight must be 1D
            sample_weights = compute_sample_weight('balanced', list)
            sample_weights_reshape = sample_weights.reshape(n, h, w, y)
            return sample_weights_reshape

def convert(seconds):
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hours, minutes, seconds)

def generate_samples(model, samples, n_classes):
    # generate fake instance
    X = model.predict(samples)  # 5 256 256 3
    print("first X", X, X.shape)
    X = np.argmax(X, axis=-1)  # 5 256 256
    print("second X", X, X.shape)
    print("unique argmax", np.unique(X))
    X = np.expand_dims(X, axis=-1)
    X = to_categorical(X, n_classes)
    print("Third X", X)
    print("Unique X expaded", np.unique(X))
    # create 'fake' class labels (0)
    return X

def summarize_performance(iteration, model, dataset_val, n_samples=5, n_classes=n_classes):
    # select a sample of input images
    [X_realA, X_realB, not_used_here] = generate_real_samples(dataset_val, n_samples)
    print("X_realA", X_realA.shape)
    # generate a batch of fake samples
    X_generated = generate_samples(model, X_realA, n_classes)  # Prediction on samples (X_realA) using model (model)
    # scale all pixels from [-1,1] to [0,1]
    # X_realA = (X_realA + 1) / 2.0 #Rescaling for plots
    # X_realB = (X_realB + 1) / 2.0 #Rescaling for plots
    # X_generated = (X_generated + 1) / 2.0 #Rescaling for plots
    #### plot real source images
    # n_samples = len(X_generated)
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)  # pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        print("Input data", X_realA[i])
        print("Shape of input", X_realA.shape)
        pyplot.imshow(X_realA[i])
        pyplot.title("input")
    #### plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)  # pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        print("Predicted data", X_generated[i])
        print("Shape of generated", X_generated.shape)
        pyplot.imshow(X_generated[i])
        pyplot.title("generated")
    ###plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)  # pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)
        pyplot.axis('off')
        print("Output data", X_realB[i])
        print("Shape of output", X_realB.shape)
        pyplot.imshow(X_realB[i])
        pyplot.title("output")
    # save a test plot to file
    filename1 = 'plot%06d.png' % (iteration + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename1 = 'model_%06d.h5' % (iteration + 1)
    model.save(filename1)
    print('>Saved: %s' % (filename1))

def generate_real_samples(dataset, n_samples, label_mask_path):
    sample_weights_reshape=create_sample_weights(label_mask_path)
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    sample_weights_batch = sample_weights_reshape[ix]
    # generate 'real' class labels (1)
    # y = ones((n_samples, patch_shape, patch_shape, 1))  ### Generate ones for every real patch
    return [X1, X2, sample_weights_batch]


def train(dataset, dataset_val, model, label_mask_path, n_epochs=EPOCHS, n_batch=BATCH_SIZE):
    print('##### Network Information ##### '
          '\n Epochs: %s \n Batch Size: %s \n '
          'Early Stopping Iterations: %s ' % (EPOCHS, BATCH_SIZE, earlyStoppingVar))

    t = time.time()
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs

    val_loss_arr = []
    g_loss_arr = []
    Xdimensions = []

    best_val_loss = 1000  # Start high, we want to end with low val loss
    valCount = 0

    g_loss_array = []
    val_loss_array = []
    n_batch_array = []

    j = 1

    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB, sample_weights_batch] = generate_real_samples(dataset, n_batch, label_mask_path)

        g_loss = model.train_on_batch(x=X_realA,
                                      y=X_realB,
                                      sample_weight=sample_weights_batch)
        g_loss_array.append(g_loss)

        [X_realA_val, X_realB_val, sample_weights_batch] = generate_real_samples(dataset_val, n_batch, label_mask_path)

        val_loss = model.test_on_batch(x=X_realA_val,
                                       y=X_realB_val,
                                       sample_weight=sample_weights_batch)
        val_loss_array.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # print('found better validation loss:' + ' ' + str(best_val_loss))
            valCount = 1

        if i == 1:
            val_loss_arr.append(val_loss)
            g_loss_arr.append(g_loss)
            Xdimensions.append(j)
            j = j + 1

        if valCount % (bat_per_epo * earlyStoppingVar) == 0:
            print('Validation loss has not improved for %.3f iterations  ' % earlyStoppingVar)
            summarize_performance(i, model, dataset)
            elapsedTime = time.time() - t  # calculate time between now and start of training
            elapsedTime = convert(elapsedTime)
            print('Elapsed training time: %s ' % elapsedTime)

            # Plot of loss vs batches at the end
            pyplot.plot(Xdimensions, g_loss_arr, color='blue', label='train loss')
            pyplot.plot(Xdimensions, val_loss_arr, color='red', label='val loss')
            pyplot.legend(loc='best')
            pyplot.ylabel('LOSS')
            pyplot.xlabel('EPOCHS')
            filename_loss = 'LossCurve.png'
            pyplot.savefig(filename_loss)
            pyplot.close()
            break

        n_batch_array.append(i + 1)
        print('>%d, train[%.3f] val[%.3f]' % (i + 1, g_loss, val_loss))

        if (i + 1) % (bat_per_epo) == 0:  # Summarize performance for every epoch
            val_loss_arr.append(val_loss)
            g_loss_arr.append(g_loss)
            Xdimensions.append(j)
            j = j + 1

            summarize_performance(i, model, dataset)

        if (i + 1) % (n_steps) == 0:  # Everything is done
            # Plot of loss vs batches at the end
            pyplot.plot(Xdimensions, g_loss_arr, color='blue', label='train loss')
            pyplot.plot(Xdimensions, val_loss_arr, color='red', label='val loss')
            pyplot.legend(loc='best')
            pyplot.ylabel('LOSS')
            pyplot.xlabel('EPOCHS')
            filename_loss = 'LossCurve.png'
            pyplot.savefig(filename_loss)
            pyplot.close()

            elapsedTime = time.time() - t  # calculate time between now and start of training
            elapsedTime = convert(elapsedTime)
            print('Elapsed training time: %s ' % elapsedTime)

        valCount = valCount + 1
        i = i + 1



# load dataset
#dataset = pd.read_csv('C:/PhD/Courses/MLOPS-Course/data/WineQT.csv')


# Preprocessing: Handle missing values if any, and select features and target
#df = dataset.dropna()

# Assume the target variable is 'quality' and features are the rest
#X = df.drop('quality', axis=1)
#y = df['quality']
#
## Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#
#
## Standardize the features
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
#
#
#
#
#
#
#
## Create an instance of a PandasDataset
#dataset = mlflow.data.from_pandas(
#    df, source='C:/PhD/Courses/MLOPS-Course/data/WineQT.csv', name="wine quality", targets="quality"
#)

#mlflow.set_tracking_uri("http://127.0.0.1:5000")
#
#
#
## Get the experiment named "Default"
#experiment_name = "Default"
#experiment = mlflow.get_experiment_by_name(experiment_name)
#
## Get all runs from the experiment
#runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
#
#
## Define a function to train and log models
#def train_and_log_model(model, model_name):
#    with (mlflow.start_run()):
#        # Train the model
#        model.fit(X_train, y_train)
#
#        # Make predictions
#        predictions = model.predict(X_test)
#
#        # Calculate metrics
#        mse = mean_squared_error(y_test, predictions)
#        r2 = r2_score(y_test, predictions)
#
#        # Log parameters, metrics, and model
#        mlflow.log_param("model_name", model_name)
#        mlflow.log_metric("mse", mse)
#        mlflow.log_metric("r2", r2)
#        model_info = mlflow.sklearn.log_model(model,
#                                              artifact_path=f"models/{model_name}")
#
#        # Create and log a plot of predictions vs. actual values
#        plt.figure(figsize=(10, 6))
#        plt.scatter(y_test, predictions, alpha=0.5)
#        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
#        plt.title(f'{model_name} Predictions vs Actual')
#        plt.xlabel('Actual Values')
#        plt.ylabel('Predicted Values')
#        plot_path = f"{model_name}_predictions_vs_actual.png"
#        plt.savefig(plot_path)
#        plt.close()
#
#        # Log the plot as an artifact
#        mlflow.log_artifact(plot_path)
#        print(f"{model_name}: MSE={mse}, R2={r2}")
#
#        # Log the Dataset to an MLflow run by using the `log_input` API
#        mlflow.log_input(dataset, context="training")
#        return model_info
#
#
## Train and log a Linear Regression model
#linear_reg_model = LinearRegression()
#model_info_lin = train_and_log_model(linear_reg_model, "LinearRegression")
#
## Train and log a Decision Tree Regressor model
#tree_model = DecisionTreeRegressor(random_state=42)
#model_info_dec = train_and_log_model(tree_model, "DecisionTreeRegressor")


loaded_model = mlflow.sklearn.load_model(model_info_dec.model_uri)


print(loaded_model)
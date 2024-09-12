import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load dataset
dataset = pd.read_csv('C:/PhD/Courses/MLOPS-Course/data/WineQT.csv')

# Preprocessing: Handle missing values if any, and select features and target
df = dataset.dropna()

# Assume the target variable is 'quality' and features are the rest
X = df.drop('quality', axis=1)
y = df['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Create an instance of a PandasDataset
dataset = mlflow.data.from_pandas(
    df, source='C:/PhD/Courses/MLOPS-Course/data/WineQT.csv', name="wine quality", targets="quality"
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")



# Get the experiment named "Default"
experiment_name = "Default"
experiment = mlflow.get_experiment_by_name(experiment_name)

# Get all runs from the experiment
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])


# Define a function to train and log models
def train_and_log_model(model, model_name):
    with (mlflow.start_run()):
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log parameters, metrics, and model
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        model_info = mlflow.sklearn.log_model(model,
                                              artifact_path=f"models/{model_name}")

        # Create and log a plot of predictions vs. actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        plt.title(f'{model_name} Predictions vs Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plot_path = f"{model_name}_predictions_vs_actual.png"
        plt.savefig(plot_path)
        plt.close()

        # Log the plot as an artifact
        mlflow.log_artifact(plot_path)
        print(f"{model_name}: MSE={mse}, R2={r2}")

        # Log the Dataset to an MLflow run by using the `log_input` API
        mlflow.log_input(dataset, context="training")
        return model_info


# Train and log a Linear Regression model
linear_reg_model = LinearRegression()
model_info_lin = train_and_log_model(linear_reg_model, "LinearRegression")

# Train and log a Decision Tree Regressor model
tree_model = DecisionTreeRegressor(random_state=42)
model_info_dec = train_and_log_model(tree_model, "DecisionTreeRegressor")


loaded_model = mlflow.sklearn.load_model(model_info_dec.model_uri)


print(loaded_model)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
import joblib
from multiprocessing import Pool

# Function to load train and test data
def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

# Function to preprocess data
def preprocess_data(train_data, test_data):
    # Encode categorical variables
    label_encode(train_data)
    label_encode(test_data)

    # Drop unnecessary columns
    train_data.drop(['num_outbound_cmds'], axis=1, inplace=True)
    test_data.drop(['num_outbound_cmds'], axis=1, inplace=True)

    return train_data, test_data

# Function to encode categorical variables
def label_encode(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])

# Function to select features
def select_features(train_data):
    X_train = train_data.drop(['class'], axis=1)
    Y_train = train_data['class']

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train, Y_train

# Function to train Random Forest Regression model and return the test score
def train_model(args):
    X_train, Y_train, params = args

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=2)

    # Random Forest Regression
    if params:
        rfr = RandomForestRegressor(**params)
    else:
        rfr = RandomForestRegressor()
    rfr.fit(x_train, y_train)

    # Evaluate model on test data
    y_pred = rfr.predict(x_test)
    score = r2_score(y_test, y_pred)

    return score

# Function to perform hyperparameter tuning using Grid Search
def grid_search(X_train, Y_train, n_jobs=-1):
    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create a Random Forest Regressor
    rfr = RandomForestRegressor()

    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=5, scoring='r2', n_jobs=n_jobs)

    # Fit the Grid Search to the data
    grid_search.fit(X_train, Y_train)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_params, best_score

# Function to perform hyperparameter tuning using Random Search
def random_search(X_train, Y_train, n_jobs=-1):
    # Define the parameter distributions to sample from
    param_dist = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create a Random Forest Regressor
    rfr = RandomForestRegressor()

    # Perform Randomized Search with cross-validation
    random_search = RandomizedSearchCV(estimator=rfr, param_distributions=param_dist, n_iter=10, cv=5, scoring='r2', random_state=42, n_jobs=n_jobs)

    # Fit the Randomized Search to the data
    random_search.fit(X_train, Y_train)

    # Get the best parameters and best score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    return best_params, best_score

# Main function
def main():
    # File paths
    train_file = 'Dataset/Train_data.csv'
    test_file = 'Dataset/Test_data.csv'

    # Number of times to train the data
    num_iterations = 5

    # Load data
    train_data, test_data = load_data(train_file, test_file)

    # Preprocess data
    train_data, test_data = preprocess_data(train_data, test_data)

    # Select features
    X_train, Y_train = select_features(train_data)

    # Create a pool of worker processes
    pool = Pool()

    # Repeat the training process multiple times
    for i in range(num_iterations):
        print(f"Iteration {i+1}:")

        # Perform hyperparameter tuning
        print("Performing Grid Search...")
        best_params_grid, best_score_grid = grid_search(X_train, Y_train, n_jobs=-1)
        print("Best Parameters (Grid Search):", best_params_grid)
        print("Best Score (Grid Search):", best_score_grid)

        print()

        print("Performing Random Search...")
        best_params_random, best_score_random = random_search(X_train, Y_train, n_jobs=-1)
        print("Best Parameters (Random Search):", best_params_random)
        print("Best Score (Random Search):", best_score_random)
        print()

        # Train model and get the test score
        print("Training model with best parameters found in Grid Search...")
        args = [(X_train, Y_train, best_params_grid)] * 5
        scores_grid = pool.map(train_model, args)
        score_grid = sum(scores_grid) / len(scores_grid)
        print("Test Score (R^2) with best parameters from Grid Search:", score_grid)
        print()

        print("Training model with best parameters found in Random Search...")
        args = [(X_train, Y_train, best_params_random)] * 5
        scores_random = pool.map(train_model, args)
        score_random = sum(scores_random) / len(scores_random)
        print("Test Score (R^2) with best parameters from Random Search:", score_random)
        print()

    # Close the pool and wait for tasks to finish
    pool.close()
    pool.join()

# Entry point of the script
if __name__ == "__main__":
    main()
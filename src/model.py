import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from datetime import date
# Rich library for better print
from rich.progress import Progress, TextColumn, SpinnerColumn
from rich import print as pprint
from rich.table import Table
from utils.df_to_rich_table import df_to_table


def parse_args():
    parser = argparse.ArgumentParser(description='Model selection for Titanic competition')
    parser.add_argument('--model', type=str, default='random_forest', choices=['rf', 'svm', 'knn', 'xgboost', 'all'],
                        help='Model to use for predictions')
    return parser.parse_args()

def load_data():
    # Load data
    path_data = "../data/"
    path_processed_data = path_data + "processed_data/"
    path_models_predictions = path_data + "models_predictions/"

    train_data = pd.read_csv(path_processed_data + "train.csv")
    X_test = pd.read_csv(path_processed_data + "test.csv")

    # We'll drop the "PassengerId" column, because it's not a feature
    X_train = train_data.drop("PassengerId", axis=1)
    #X_test = test_data.drop("PassengerId", axis=1)

    # Remove the target variable from the features
    X_train = X_train.drop("Survived", axis=1)
    y_train = train_data["Survived"]

    return X_train, y_train, X_test, path_models_predictions

def saving_best_params(best_params, best_score, model_name, path_models_predictions):
    path_models_scores = path_models_predictions+ "/scores/"
    with open(path_models_scores + model_name + "_best_score.txt", "a") as file:
        file.write(f"=== {date.today()} ===\n")
        file.write(f"Best parameters: {best_params}\n")
        file.write(f"Best score: {best_score}\n")

def grid_search(model, parameters, model_name=""):
    spinner_column = SpinnerColumn(finished_text="Grid Search done!")
    text_column = TextColumn(f"[bold cyan]Grid Search for {model_name}[/bold cyan]...")
    progress = Progress(spinner_column, text_column, transient=True)
    with progress:
        task = progress.add_task("")
        # Grid Search to found the best hyperparameters
        rdf_clf = GridSearchCV(estimator=model, param_grid=parameters) # model grid search
        rdf_clf.fit(X_train, y_train)

    # Grid Search Results
    pprint(f"[bold cyan]{model_name} Model Metrics[/bold cyan]")
    print("Best parameters:", rdf_clf.best_params_)
    print("Best score:", rdf_clf.best_score_)

    # saving the best parameters and best score in a txt file
    saving_best_params(best_params=rdf_clf.best_params_, best_score=rdf_clf.best_score_,
                        model_name=model_name, path_models_predictions=path_models_predictions)

    return rdf_clf.best_estimator_

def predict_and_save(model, X_test, path_models_predictions, model_name=""):
    
    # remove the "PassengerId" column
    passenger_id = X_test["PassengerId"]
    X_test = X_test.drop("PassengerId", axis=1)
    
    y_test = model.predict(X_test)
    y_test = pd.DataFrame({"PassengerId":passenger_id, "Survived":y_test})
    y_test.to_csv(path_models_predictions + model_name + "_prediction.csv", index=False) 

    return y_test



if __name__ == "__main__":
    args = parse_args()

    all_models = False
    if args.model == "all":
        all_models = True
 
    # Load data
    X_train, y_train, X_test, path_models_predictions = load_data()
 
    
    if args.model == "rf" or all_models:
        # ============= Random Forest Model =============
        # Defining the Model
        rdf_model = RandomForestClassifier(random_state=42)

        # Grid Search 
        parameters = {'n_estimators':[10, 50, 100, 500], 'max_depth':[5, 10, 20]} # hyperparameters to test
        rdf_model = grid_search(rdf_model, parameters, "Random Forest") 

        # Do predictions and save the results
        y_test = predict_and_save(rdf_model, X_test, path_models_predictions, "RandomForest")

        # Random Forest feature importances
        feature_importances = rdf_model.feature_importances_
        feature_importances = pd.DataFrame({"Feature":X_train.columns, "Importance":feature_importances})
        feature_importances = feature_importances.sort_values(by="Importance", ascending=False)
        print("Random Forest Feature Importances:")

        # Initiate a Table instance to be modified
        table = Table(show_header=True, header_style="bold magenta")
        feature_importance_table = df_to_table(feature_importances, table)
        pprint(feature_importance_table)

    # ============= Support Vector Machine Model =============
    if args.model == "svm" or all_models: 
        # Defining the Model
        svm_model = SVC()

        # Grid search
        parameters = {'C': [0.5, 1, 2], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
        svm_model = grid_search(svm_model, parameters, "SVM")

        # Do predictions and save the results
        y_test = predict_and_save(svm_model, X_test, path_models_predictions, "SVM")   

    # ============= K-Nearest Neighbors Model =============
    if args.model == "knn" or all_models:
        # Defining the Model
        knn_model = KNeighborsClassifier()

        # Grid search
        parameters = {'n_neighbors': [3, 5, 7, 10, 20, 50, 100], 'weights': ['uniform', 'distance']}
        knn_model = grid_search(knn_model, parameters, "KNN")

        # Do predictions and save the results
        y_test = predict_and_save(knn_model, X_test, path_models_predictions, "KNN")

    # ============= XGBoost Model =============    
    if args.model == 'xgboost' or all_models:
        # Defining the Model
        xgb_model = XGBClassifier()

        # Grid search
        parameters = {'max_depth': [3, 6, 10], 
                      'gamma': [0, 0.01, 0.1],
                      'learning_rate': [0.1, 0.01],
                      'lambda':[0.1, 1, 10],
                      'scale_pos_weight': [1, 3, 5],
                      'tree_method':['exact']}
        
        xgb_model = grid_search(xgb_model, parameters, "XGBoost")

        # Do predictions and save the results
        y_test = predict_and_save(xgb_model, X_test, path_models_predictions, "XGBoost")
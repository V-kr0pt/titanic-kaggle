import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def load_data():
    # Load data
    path_data = "../data/"
    path_processed_data = path_data + "processed_data/"
    path_models_predictions = path_data + "models_predictions/"

    train_data = pd.read_csv(path_processed_data + "train.csv")
    X_test = pd.read_csv(path_processed_data + "test.csv")

    X_train = train_data.drop("Survived", axis=1)
    y_train = train_data["Survived"]

    return X_train, y_train, X_test, path_models_predictions


def grid_search(model, parameters, model_name=""):   
    # Grid Search to found the best hyperparameters
    rdf_clf = GridSearchCV(estimator=model, param_grid=parameters) # model grid search
    rdf_clf.fit(X_train, y_train)

    # Grid Search Results
    print(f"{model_name} Model Metrics:")
    print("Best parameters:", rdf_clf.best_params_)
    print("Best score:", rdf_clf.best_score_)

    return rdf_clf.best_estimator_

def predict_and_save(model, X_test, path_models_predictions, model_name=""):
    y_test = model.predict(X_test)
    y_test = pd.DataFrame({"PassengerId":X_test["PassengerId"], "Survived":y_test})
    y_test.to_csv(path_models_predictions + model_name + "_prediction.csv", index=False) 

    return y_test



if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, path_models_predictions = load_data()
 
    # Random Forest Model
    rdf_model = RandomForestClassifier(random_state=42)
    
    # Grid Search 
    parameters = {'n_estimators':[10, 50, 100, 500], 'max_depth':[5, 10, 20]} # hyperparameters to test
    rdf_model = grid_search(rdf_model, parameters, "Random Forest") 
    y_test = predict_and_save(rdf_model, X_test, path_models_predictions, "RandomForest")

    # Random Forest feature importances
    feature_importances = rdf_model.feature_importances_

    #rdf_model.fit(train_data.drop("Survived", axis=1), train_data["Survived"])



    
    
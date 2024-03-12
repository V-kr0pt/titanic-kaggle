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


if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, path_models_predictions = load_data()
 
    # Random Forest Model
    rdf_model = RandomForestClassifier(random_state=42)
    
    # Grid Search to found the best hyperparameters
    parameters = {'n_estimators':[10, 50, 100, 500], 'max_depth':[5, 10, 20]}
    rdf_clf = GridSearchCV(estimator=rdf_model, param_grid=parameters) # radom forest classifier grid search
    rdf_clf.fit(X_train, y_train)

    # Grid Search Results
    print("Random Forest Model Metrics:")
    print("Best parameters:", rdf_clf.best_params_)
    print("Best score:", rdf_clf.best_score_)
    
    # load the best model
    rdf_model = rdf_clf.best_estimator_
    y_test = rdf_model.predict(X_test)

    rdf_y_test = pd.DataFrame({"PassengerId":X_test["PassengerId"], "Survived":y_test})
    rdf_y_test.to_csv(path_models_predictions + "RF_prediction.csv", index=False) 

    # Random Forest feature importances
    feature_importances = rdf_model.feature_importances_

    #rdf_model.fit(train_data.drop("Survived", axis=1), train_data["Survived"])



    
    
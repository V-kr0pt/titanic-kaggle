import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


if __name__ == "__main__":
    # Load data
    path_data = "../data/"
    path_processed_data = path_data + "processed_data/"
    path_models_predictions = path_data + "models_predictions/"

    train_data = pd.read_csv(path_processed_data + "train.csv")
    X_test = pd.read_csv(path_processed_data + "test.csv")

    X_train = train_data.drop("Survived", axis=1)
    y_train = train_data["Survived"]
 
    # Random Forest Model
    rdf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    rdf_model.fit(train_data.drop("Survived", axis=1), train_data["Survived"])
    y_test = rdf_model.predict(X_test)

    rdf_y_test = pd.DataFrame({"PassengerId":X_test["PassengerId"], "Survived":y_test})

    rdf_y_test.to_csv(path_models_predictions + "RF_prediction.csv", index=False)


    
    
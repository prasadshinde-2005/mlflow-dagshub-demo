
import mlflow
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os


import dagshub
dagshub.init(repo_owner='prasadshinde-2005', repo_name='mlflow-dagshub-demo', mlflow=True)
print("Tracking URI:", mlflow.get_tracking_uri())

mlflow.set_tracking_uri(
    "https://dagshub.com/prasadshinde-2005/mlflow-dagshub-demo.mlflow"
)


# for using mlrun
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

mlflow.set_tracking_uri("file:./mlruns")



# import iris 
iris = load_iris()
X = iris.data
y = iris.target

# sliting data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parmeters 
max_depth = 10



mlflow.set_experiment("iris_decision_tree")

with mlflow.start_run(run_name ="sp-exp_plot"):
    DT = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    DT.fit(X_train, y_train)
    y_pred = DT.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test ,y_pred)

    mlflow.log_metric("accuracy", accuracy)
    
    mlflow.log_param("max_depth", max_depth)
  
    print("Accuracy:", accuracy)
    print("confusion_matrix" ,cm)

    # creat plot for confusion matrix 
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # ---- Image file save karo ----
    plt.savefig("confusion_matrix.png")
    plt.close()  # memory leak avoid karne ke liye band karo

    # ---- MLflow ko artifact ke roop mein log karo ----
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(DT , "decision_tree", serialization_format ="pickle")
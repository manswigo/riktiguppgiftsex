import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
# Load the test data
test_df = pd.read_csv("grid_stability(in).csv")
X_test = test_df.drop(["stabf", "stab"], axis=1)
y_test = test_df["stabf"]
try:
    with open("model_classification.pkl", "rb") as f:
        model = pickle.load(f)
# Make predictions
    predictions = model.predict(X_test)
# Compute accuracy
    accuracy = accuracy_score(y_test, predictions)
    print("Baseline accuracy:",accuracy)
    print()
    print("Classification report:")
    print(classification_report(y_test, predictions))
    print("confusion matrix")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    if accuracy > 0.75:
        print(f"[PASS] Model passed the test with accuracy {accuracy:.2f}!")
    else:
        print(f"[FAIL] Model failed the test with accuracy {accuracy:.2f}!")
except Exception as e:
    print(f"[FAIL] Model failed the test with error: {e}")
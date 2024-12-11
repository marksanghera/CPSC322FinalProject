# train_model.py
import pickle
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.mypytable import MyPyTable
from sklearn.model_selection import train_test_split

discretized_table = MyPyTable()
discretized_table.load_from_file("/home/FinalProject/neo_discretized.csv")

print("Data loaded successfully")
print(f"Number of samples: {len(discretized_table.data)}")
print(f"Features: {discretized_table.column_names}")

# Prepare X and y
X = [row[:-1] for row in discretized_table.data]
y = [row[-1] for row in discretized_table.data]

dt_classifier = MyDecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
dt_classifier.fit(X_train, y_train)

# Verify the model works before saving
test_example = X_test[0]
print("\nTesting model before saving...")
print(f"Test input: {test_example}")
test_prediction = dt_classifier.predict([test_example])
print(f"Prediction: {test_prediction}")

print("\nSaving model...")
with open('model.pkl', 'wb') as f:
    pickle.dump(dt_classifier, f)

print("Verifying saved model...")
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    verification_prediction = loaded_model.predict([X_test[0]])
    print(f"Verification prediction: {verification_prediction}")

print("\nModel trained and saved successfully!")
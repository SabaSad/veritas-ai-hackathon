"""
Environment smoke test for the hackathon setup.
Not part of the actual project -- just confirms the venv and packages work
by training a throwaway model on scikit-learn's built-in dataset.
"""

import importlib
import importlib.metadata as metadata

# Maps the requirements.txt distribution name to its importable module name
# (some packages are installed under one name but imported under another).
PACKAGES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("jupyter", "jupyter"),
    ("ipykernel", "ipykernel"),
]

print("=== Package versions ===")
for dist_name, module_name in PACKAGES:
    importlib.import_module(module_name)  # confirms it's actually importable
    version = metadata.version(dist_name)
    print(f"{dist_name}: {version}")

print("\n=== Training smoke test (logistic regression on breast cancer dataset) ===")

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, stratify=data.target, random_state=42
)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
matrix = confusion_matrix(y_test, predictions)

print(f"Test accuracy: {accuracy:.4f}")
print("Confusion matrix:")
print(matrix)

print("\nSetup check complete -- environment is working.")

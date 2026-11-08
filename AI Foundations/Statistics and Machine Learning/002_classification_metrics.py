from sklearn.metrics import classification_report, confusion_matrix


actual = ["billing", "access", "bug", "billing", "access", "bug"]
predicted = ["billing", "access", "billing", "billing", "bug", "bug"]
labels = ["access", "billing", "bug"]

print("Confusion matrix (rows are actual, columns are predicted):")
print(confusion_matrix(actual, predicted, labels=labels))
print("\nPer-category precision, recall, and F1:")
print(classification_report(actual, predicted, labels=labels, zero_division=0))

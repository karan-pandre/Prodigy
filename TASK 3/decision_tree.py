
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/karanpandre/Documents/INTERNSHIP/Prodigy/TASK 3/bank-additional-full.csv', sep=';')


print("First few rows of the dataset:")
print(df.head())
print()


print("Shape of the dataset:")
print(df.shape)
print()


print("Checking for missing values:")
print(df.isnull().sum())
print()


print("Data types:")
print(df.dtypes)
print()


df = pd.get_dummies(df, drop_first=True)


X = df.drop('y_yes', axis=1)  
y = df['y_yes'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
print()


clf = DecisionTreeClassifier(random_state=42)


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print()


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()


print("Classification Report:")
print(classification_report(y_test, y_pred))
print()


plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.title("Decision Tree Classifier - Bank Marketing Dataset")
plt.show()

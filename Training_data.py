import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open("data.pickle", "rb"))
data = np.asarray(data_dict["data"])
alphabets = np.asarray(data_dict["labels"])

X_train, X_test, y_train, y_test = train_test_split(data, alphabets, test_size=0.2, shuffle=True)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_pred, y_test))

f = open("model.pickle", "wb")
pickle.dump({"model": rf}, f)
f.close()


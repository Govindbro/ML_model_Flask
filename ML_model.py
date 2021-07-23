import pandas as pd
import numpy as np
import pickle # for saving my model
#reading the iris dataset [That are in string format]
df = pd.read_csv('iris.data')

X = np.array(df.iloc[:, 0:4]) # first 4 columns
y = np.array(df.iloc[:, 4:]) # last column
#Now our ML model usually works on integer dataset : so we have to covnert our data into integer
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # to convert string data into integer
y = le.fit_transform(y.reshape(-1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(X_train,y_train)


pickle.dump(sv, open('iri.pkl', 'wb'))

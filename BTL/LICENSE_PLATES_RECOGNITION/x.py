import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pickle


df = pd.DataFrame([1,2,3,4])

print(df)
print(df.T)
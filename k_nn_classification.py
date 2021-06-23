
#This is an edit for github
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer

df = pd.read_csv('Country-data.csv') # load dataframe here
infile = open('finalized_model.pickle', 'rb') # load model here
model = pickle.load(infile)
infile.close()

class model_predict:

    def __init__(self, df, model):
        df = df.drop(['country','child_mort','income'], 1)
        self.df = df
        self.model = model


    def preprocess(self):
        X = self.df

        scaler = QuantileTransformer(output_distribution='normal')
        scaler.fit(X)

        scaled_numerical_data = scaler.transform(X)  # standardising numerical cols
        X_final = pd.DataFrame(scaled_numerical_data, index=X.index,
                               columns=X.columns)  # create df of standardised numerical cols
        return X_final

    def predict_result(self,X):
        model = self.model
        result = model.predict(X)
        return result

predictor = model_predict(df,model)
cleaned_df = predictor.preprocess()
print(cleaned_df.head())
result = predictor.predict_result(cleaned_df)
print(result)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import stats as st
from sklearn import metrics
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class missingVals():

    def numerical(self, df):

        #separates numeric columns
        df_numeric = df._get_numeric_data()
        df_numeric.info()


        if self.option=='auto':
            #adds 2 extra columns with mean and median, deletes original column
            for col in df_numeric.columns:
                if df_numeric[col].isnull().sum() > 0:
                    df_numeric[str(col) + '_mean'] = df_numeric[col].fillna(df_numeric[col].mean()) 
                    df_numeric[str(col) + '_median'] = df_numeric[col].fillna(df_numeric[col].median())
                    df_numeric.dropna(axis='columns')

        elif self.option=='delete':
            #drops missing values from dataset
            for col in df_numeric.columns:
                if df_numeric[col].isnull().sum() > 0:
                    df_numeric.dropna(inplace=True) #drops rows with missing values
                    df_numeric.reset_index(drop=True)

        elif self.option=='logreg':
            #predicts missing values using linear regression algorithm
            for col in df_numeric.columns:
                if df_numeric[col].isnull().sum() > 0:
                    y1=df_numeric[col]
                    X_train, X_test,y_train,y_test = train_test_split(updated_df,y1,test_size=0.3)
                    lr = LogisitcRegression()
                    lr.fit(X_train,y_train)
                    pred = lr.predict(X_test)
                    print(metrics.accuracy_score(pred,y_test))
     
        elif self.option=='knn':
            #predicts missing values using knn
            imputer = KNNImputer(n_neighbors=5)
            imputer.fit_transform(df_numeric)

        elif self.option=='mean':
            #replaces missing values with mean
            for col in df_numeric.columns:
                if df_numeric[col].isnull().sum() > 0:
                    df_numeric[col].fillna(df.numeric[col].mean())

        elif self.option=='median':
            #replaces missing values with median
            for col in df_numeric.columns:
                if df_numeric[col].isnull().sum() > 0:
                    df_numeric[col].fillna(df.numeric[col].median())

        elif self.option=='mode':
            #replaces missing values with mode
            for col in df_numeric.columns:
                if df_numeric[col].isnull().sum() > 0:
                    df_numeric[col].fillna(df.numeric[col].mode())

        elif self.option=='linreg':
            #predicts missing values with linear regression
            for col in df_numeric.columns:
                if df_numeric[col].isnull().sum() > 0:
                    lr = LinearRegression()
                    testdf = df_numeric[df_numeric[col].isnull()==True]
                    traindf = df_numeric[df_numeric[col].isnull()==False]
                    y = traindf[col]
                    traindf.drop(col,axis=1,inplace=True)
                    lr.fit(traindf,y)
                    testdf.drop(col,axis=1,inplace=True)
                    pred = lr.predict(testdf)
                    testdf[col]= pred
                    traindf[col]=y
                    
                       
    def categorical_values(self,df):
            cols = df.columns
            num_cols = df._get_numeric_data().columns
            cat_cols = list(set(cols) — set(num_cols))               

            imp = SimpleImputer(strategy="most_frequent")
            df_cat = pd.DataFrame(imp.fit_transform(df_cat), columns=cat_cols)
            
            df_label= df_cat.apply(LabelEncoder().fit_transform)
            
            ohe = OneHotEncoder()
            ohe.fit(df_label[cat_cols])
            df_ohe = ohe.transform(df_label[cat_cols])
            
            df_cat = pd.DataFrame(df_ohe.toarray())
            
                       
                    
           


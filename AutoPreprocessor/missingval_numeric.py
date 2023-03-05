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


df = pd.read_csv(dataset)
print (df.isnull().sum())
df.info()


class Missing_Values_Numerical(self):

    def numerical(self, df, option, paramaxis):

        #separates numeric columns
        df_numeric = df._get_numeric_data()
        df_numeric.info()


        if option=='auto':
            #adds 2 extra columns with mean and median, deletes original column
            for col in df_numeric.columns:
                if df_numeric[col].isnull().sum() > 0:
                    df_numeric[str(col) + '_mean'] = df_numeric[col].fillna(df_numeric[col].mean()) 
                    df_numeric[str(col) + '_median'] = df_numeric[col].fillna(df_numeric[col].median())
                    df_numeric.dropna(axis='columns')

        elif option=='delete':
            #drops missing values from dataset
            for col in df_numeric.columns:
                if df_numeric[col].isnull().sum() > 0:
                    df_numeric.dropna(inplace=True) #drops rows with missing values
                    df_numeric.reset_index(drop=True)

        elif option=='logreg':
            #predicts missing values using linear regression algorithm
            for col in df_numeric.columns:
                if df_numeric[col].isnull().sum() > 0:
                    y1=updated_df[col]
                    X_train, X_test,y_train,y_test = train_test_split(updated_df,y1,test_size=0.3)
                    lr = LogisitcRegression()
                    lr.fit(X_train,y_train)
                    pred = lr.predict(X_test)
                    print(metrics.accuracy_score(pred,y_test))
     
        elif option=='knn':
            #predicts missing values using knn
            imputer = KNNImputer(n_neighbors=5)
            imputer.fit_transform(df_numeric)

        elif option=='mean':
            #replaces missing values with mean
            for col in df_numeric.columns:
                if df_numeric[col].isnull().sum() > 0:
                    df_numeric[col].fillna(df.numeric[col].mean())

        elif option=='median':
            #replaces missing values with median
            for col in df_numeric.columns:
                if df_numeric[col].isnull().sum() > 0:
                    df_numeric[col].fillna(df.numeric[col].median())

        elif option=='mode':
            #replaces missing values with mode
            for col in df_numeric.columns:
                if df_numeric[col].isnull().sum() > 0:
                    df_numeric[col].fillna(df.numeric[col].mode())

        elif option=='linreg':
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




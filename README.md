# Auto-Preprocessor
Data preprocessing is an essential aspect of any machine learning project, and our aim is to maximise automation in this aspect. The goal of this project is to create a python project which effectively modifies and cleans up the dataset in a manner which enhances the performance of the algorithms. Duplicates within the dataset are dropped and NaN values are either dropped or fixed based on user choice. We then use feature engineering to identify the main correlated factors within the dataset which allows the user to make predictions.

#Installation
pip install -i https://test.pypi.org/simple/ AutoPreprocessor==0.0.1

#Syntax
Autopreprocess(self, input_data, missingVal = "auto", duplicates=False, bool=False, extract_datetime=False, outliers=False, outlier_param=1.5)


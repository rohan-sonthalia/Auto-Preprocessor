import pandas as pd
from AutoPreprocessor import boolean
from AutoPreprocessor import dateTime
from AutoPreprocessor import duplicates
from AutoPreprocessor import missingVals
from AutoPreprocessor import outliers

class AutoPreProcess:

    def __init__(self, input_data, missingVal = "auto", duplicates=False, bool=False, extract_datetime=False, outliers=False, outlier_param=1.5):        
        output_data = input_data.copy()

        self.duplicates = duplicates
        self.outliers = outliers
        self.extract_datetime = extract_datetime
        self.outlier_param = outlier_param
        self.bool = bool
        self.missingVal = missingVal

        # initialize our class and start the autoclean process
        self.output = self._clean_data(output_data, input_data)
            
    def _clean_data(self, df):
        df = df.reset_index(drop=True)
        df = missingVals.handle(self, df)
        if self.duplicates:
            df = duplicates.handle(self, df)
        if self.outliers:
            df = outliers.handle(self, df)
        if self.extract_datetime:    
            df = dateTime.convert_datetime(self, df)
        if self.bool:
            df = boolean.handle(self,df)
        return df 
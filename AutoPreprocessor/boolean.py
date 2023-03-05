import pandas as pd
import numpy as np

class boolean:

    def convert(self, df):
        if self.boolValues:
            cols = df.select_dtypes(include='bool').columns
            for feature in cols:
                df[feature].astype(int)
            return df
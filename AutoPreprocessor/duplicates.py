import pandas as pd

class duplicates:

    def handle(self, df):
        if self.duplicates:
            df.drop_duplicates(inplace=True, ignore_index=False)
            df = df.reset_index(drop=True)
        return df 
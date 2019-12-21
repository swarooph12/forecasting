import os
import settings
import pandas as pd
import numpy as np

def merge():
    ''' merge sales and article data and save it for later use'''
    
    #load txt file containing the sales
    types = {'country': np.dtype(str)}
    sales = pd.read_csv(os.path.join('data', "sales.txt"),sep = ';', parse_dates=[6] ,dtype=types)
    
    #load txt file containg article
    articles = pd.read_csv(os.path.join('data', "article_attributes.txt"),sep = ';',dtype=types)

    #merge dataframes on article id number and save in processed_dir
    sales = pd.merge(sales, articles, on='article')
    sales.to_csv(os.path.join(settings.PROCESSED_DIR, "{}.txt".format('sales')), index=False)
    return sales


if __name__ == "__main__":
    merge()


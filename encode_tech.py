from unittest.mock import inplace

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import sys
from sklearn.preprocessing import OneHotEncoder
from log_code import setup_logging
logger=setup_logging('encod')

def encoding(x_train_cat,x_test_cat):
    try:
        cols=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling','PaymentMethod', 'sim']
        oh=OneHotEncoder(categories='auto',drop='first',handle_unknown='ignore')
        oh.fit(x_train_cat[[cols]])
        logger.info(f'{oh.categories_}')
        logger.info(f'{oh.get_feature_names_out()}')
        res=oh.transform(x_train_cat[[cols]]).toarray()
        res1 = oh.transform(x_test_cat[[cols]]).toarray()
        f=pd.DataFrame(res,columns=oh.get_feature_names_out())
        f1= pd.DataFrame(res, columns=oh.get_feature_names_out())
        x_train_cat.rest_index(drop=True,inplace=True)
        f.reset_index(drop=True,inplace=True)
        x_test_cat.rest_index(drop=True, inplace=True)
        f1.reset_index(drop=True, inplace=True)
        x_train_cat=pd.concat([x_train_cat.drop(column=cols),f],axis=1)
        x_test_cat = pd.concat([x_test_cat.drop(column=cols), f], axis=1)
        logger.info(x_train_cat.isnull().sum())
        logger.info(x_test_cat.isnull().sum())
        logger.info(f'-------------------------------')
        logger.info(x_train_cat.sample(5))
        logger.info(f'--------------------')
        logger.info(x_test_cat.sample(5))
        return x_train_cat,x_test_cat
    except Exception as e:
        e_type,e_msg,e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')
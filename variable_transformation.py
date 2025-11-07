import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from scipy.special import lambertw
from log_code import setup_logging
logger=setup_logging('trans')
import sklearn
import sys
from scipy import stats

def quantile(x_train,x_test):
    try:
        df=x_train.copy()
        df1=x_test.copy()
        qt=QuantileTransformer(output_distribution='normal',random_state=24)
        transformed=qt.fit_transform(df)
        transform=qt.transform(df1)
        a=pd.DataFrame(transformed,columns=[i+'_qan' for i in df.columns],index=df.index)
        b=pd.DataFrame(transform,columns=[j+'_qan' for j in df1.columns],index=df1.index)
        logger.info(a.columns)
        logger.info(b.columns)
        return a,b
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')
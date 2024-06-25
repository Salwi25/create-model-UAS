# -*- coding: utf-8 -*-

import pandas as pd
import joblib
   
if __name__ == '__main__':
    filename = 'lr_joblib.model'
    loaded_model= joblib.load(filename)
   
    df_input = pd.read_csv('input_data.csv')
    result = loaded_model.predict(df_input)
    #print(result)
    
    for i in result:
        int_result = int(i)
        if(int_result==0):
            decision='Benign'
        elif (int_result==1):
            decision='Malignant'
        else:
            decision='Not defined'
    
        print('Disease is ', decision)
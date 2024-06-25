import pandas as pd
import joblib
    
if __name__ == '__main__':
    #load the model
    filename = 'lr_joblib.model'
    loaded_model= joblib.load(filename)
    #create new unlabelled test data
    df_input = pd.DataFrame(columns = ["radius_mean","texture_mean","perimeter_mean","area_mean",
                                       "smoothness_mean","compactness_mean","concavity_mean","concave points_mean",
                                       "symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se",
                                       "area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se",
                                       "fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
                                       "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"])
    df_input.loc[0] = [16.13,20.68,108.1,798.8,0.117,0.2022,0.1722,0.1028,0.2164,0.07356,0.5692,1.073,3.854,54.18,0.007026,0.02501,0.03188,0.01297,0.01689,0.004142,20.96,31.48,136.8,1315,0.1789,0.4233,0.4784,0.2073,0.3706,0.1142]
    #make prediction
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

    
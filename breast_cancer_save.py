import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

def lr(X, y):
    model = LogisticRegression(random_state=0)
    model = model.fit(X, y)
    filename = 'lr_joblib.model'
    joblib.dump(model, filename)
    
if __name__ == '__main__':
    #read the dataset
    pd.options.display.max_columns=None
    df = pd.read_csv('data.csv')
    df = df.drop(columns=['id','Unnamed: 32'])

    #get the X and y
    df_X = df.drop(['diagnosis'],axis=1)
    df_y = df['diagnosis']

    #relabelling the class
    df_y=df_y.replace('B',0)
    df_y=df_y.replace('M',1)
    
    #menyimpan X dan y menjadi numpy arrays
    X = df_X.astype(float).values
    y = df_y.astype(float)

    #hold-out method, dibagi menjadi training dan testing set. 70% training, 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #scaling
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #get the numerical attributes
    #df_X_numb = df_X.drop(columns=['diagnosis'], axis=1)
    #convert single categorical column into numeric, label encoding    
    #le = preprocessing.LabelEncoder()
    #df_X['diagnosis'] = le.fit_transform(df_X['diagnosis'])

    #generate the model 
    print('Generate Logistic Regression Model')
    lr(X_train, y_train)    
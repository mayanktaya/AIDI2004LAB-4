import pandas as pd
import numpy as np

df=pd.read_csv('fish.csv')
df

# Independent and Dependent features
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

# Train Test Split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# Implement Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

# Prediction
y_pred=classifier.predict(X_test)

# Check Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
score

import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
from flask import Flask, request, jsonify, render_template

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = classifier.predict(final_features)

    
    return render_template('index.html', prediction_text='The fish belong to species {}'.format(prediction))
    
    


if __name__=='__main__':
    app.run()
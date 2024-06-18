from flask import Flask, render_template,request,url_for,redirect
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/',methods = ["GET","POST"])
def index():

    if request.method == "GET":
        encoder = pickle.load(open("artifacts/labelencoder.pkl","rb"))
        options = [{"value": i, "text": f"{i}"} for i in encoder.classes_]
        return render_template('index.html', options=options)
    
    elif request.method == "POST":
        def encoding(loc):
            encoder = pickle.load(open("artifacts/labelencoder.pkl","rb"))
            return encoder.transform([loc])[0]
        
        def get_dataframe(area,bath,bhk,encoded_location):
            df = pd.DataFrame([{"total_sqft":area,"bath":bath,"BHK":bhk,"encoded_loc":encoded_location}])
            return df
        
        def predictor(area,bath,bhk,encoded_location):
            model = pickle.load(open("artifacts/model.pkl","rb"))
            df = get_dataframe(area,bath,bhk,encoded_location)
            return model.predict(df)[0]
            

        
        areaip = request.form['area']
        bathip = float(request.form["bathroom"])
        bhkip = int(request.form["bhk"])
        location = request.form["location"]
        encoded_locationip = encoding(location)

        prediction = str(int(predictor(areaip,bathip,bhkip,encoded_locationip)))

        def beautify(prediction):
            if len(prediction)>=3:
                return prediction[0]+ " Crores "+ prediction[1]+prediction[2]+" Lakhs"
            else:
                return prediction + " Lakhs"
        
        return render_template("index.html",output=beautify(prediction))
         

if __name__ == '__main__':
    app.run(debug=True)

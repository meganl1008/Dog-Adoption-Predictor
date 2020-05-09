# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('./model/model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    # Get the data from the POST request.
    if request.method == "POST":
        #data = request.get_json(force=True)
        print(request.form['age'])
        age_data = float(request.form['age'])
        color_data = float(request.form['color'])
        breed_data = float(request.form['breed'])
        gender_data = float(request.form['gender'])
        neutered_data = float(request.form['neutered'])
        
        print(age_data)
        print(color_data)
        print(breed_data)
        print(gender_data)
        print(neutered_data)

        # enter csv
        df = pd.read_csv('x_values.csv')
        print(df)

        X_data = [[age_data, color_data, breed_data, gender_data, neutered_data]]
        df_2 = pd.DataFrame(X_data, columns = ['age_months','color_weights', 'breed_weights', 'gender_weights', 'fixed_weights'])
        final_df = pd.concat([df, df_2])
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(final_df)
        print(X_train_scaled[-1])

        print("Data", model.predict([X_train_scaled[-1]]))

        # # Make prediction using model loaded from disk as per the data.
        prediction = model.predict([X_train_scaled[-1]])

        # # Take the first value of prediction
        output = prediction[0]

        print(prediction)

        if (output == 1): 
            output = './images/adopted.png'
            gif = 'https://media.giphy.com/media/sMaW02wUllmFi/giphy.gif'
            color = 'Green'
            background_color = 'lightgreen'
            font_color = 'darkgreen'
            width = '40%'

        else:
            output = './images/not_adopted.png'
            gif = 'https://media.giphy.com/media/12G1D7rPEV5Cfu/giphy.gif'
            color = 'Red'
            background_color = 'Grey'
            font_color = 'Black'
            width = '65%'

        return render_template("results.html", output=output, gif=gif, color=color, background_color=background_color, font_color=font_color, width=width)

if __name__ == '__main__':
    app.run(debug=True)

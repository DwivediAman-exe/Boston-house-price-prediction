import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# creating a flask app
app = Flask(__name__)

# Loading the model and scalar model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
	# parse data from request
	data = request.json['data']
	print(data)
	# Get values from data dictionary
	data_vals = data.values()
	# Convert dictionary to list
	data_vals = list(data_vals)
	# Convert to 2D array
	data_vals = np.array(data_vals).reshape(1,-1)
	print(data_vals)

	# Transform data using scalar model
	new_data = scalar.transform(data_vals)

	# Predict the output using regmodel
	output = regmodel.predict(new_data)
	print(output[0])
	# Return output in response
	return jsonify(output[0])

if __name__=="__main__":
	app.run(debug=True)
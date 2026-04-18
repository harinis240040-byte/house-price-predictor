from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
data = pd.read_csv("house_data1.csv")

# Handle missing values
data = data.fillna({
    'bathroom': 1,
    'bhk': 2
})

# Convert price from lakhs to rupees


# Features & target
X = data[['area', 'bhk', 'bathroom']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])

    prediction = model.predict([[area, bedrooms, bathrooms]])[0]

    return render_template('index.html',
                           prediction=round(prediction, 2),
                           areas=list(data['area'][:20]),
                           prices=list(data['price'][:20]))

if __name__ == '__main__':
    app.run(debug=True)

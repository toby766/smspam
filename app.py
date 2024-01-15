from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Create a Flask app
app = Flask(__name__)
cvm = pickle.load(open('vectorizer.pkl', 'rb'))
# Load the trained model
model = pickle.load(open('clf.pkl', 'rb'))

# Define a route for the home page
@app.route('/')
def home():
  return render_template('home.html')

# Define a route for the predict page
@app.route('/predict', methods=['POST'])
def predict():
  # Get the message from the request
  message = request.form.get('content')
  
  # Vectorize the message
  data = cvm.transform([message])
  
  # Make a prediction
  prediction = model.predict(data)[0]
  prediction = 'spam' if prediction == 'spam' else 'ham'
  # Return the prediction
  return render_template('home.html', prediction=prediction)

# Run the app
if __name__ == '__main__':
  app.run(debug=True)
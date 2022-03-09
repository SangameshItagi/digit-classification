from flask import Flask
from flask import Flask, render_template, request
from DigitClassifier import classifyImage
app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    if (request.files['image']): 
        file = request.files['image']
        result = classifyImage(file)
        print("Model classification: ", result)        
        return str(result)

@app.route('/')
def index():
    return "Hello from FLASK"
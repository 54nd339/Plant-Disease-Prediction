import requests
from flask import Flask, render_template, request
import base64
import os
import requests

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image'] # fet input
        app.logger.info(file)
        filename = file.filename  
        
        file_path = os.path.join('static/upload/test', filename)
        with open(file_path, 'rb') as f:
            file_data = base64.b64encode(f.read()).decode('utf-8')
        
        file_data = "data:image/jpg;base64," + file_data
        
        try:
            response = requests.post("https://art3mis011-plantdiseasedetection.hf.space/run/predict", json={
                "data": [ file_data ]
            }).json()
            output = response["data"][0]["label"]
        except Exception as e:
            print("Error: ", e)

        pred, output_page = output, output + '.html'
        return render_template(output_page, pred_output = pred, user_image = file_path)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,port=8080) 
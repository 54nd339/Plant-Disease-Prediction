import requests
from flask import Flask, render_template, request
import base64
import os
import requests
from gradio_client import Client

client = Client("https://art3mis011-new-plant-disease-detection.hf.space/--replicas/fm89j/")
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')

# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        app.logger.info(file)
        filename = file.filename        
        
        file_path = os.path.join('E:/Z My Works/Codehastra_Hackathon/Web-Deployment/static/upload/test/', filename)
        file.save(file_path)

        with open(file_path, "rb") as image_file:
            file_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Pass the image to the model
        try:
            result = client.predict(file_data, api_name="/predict")
            pred = result["label"]
            # output_page = result["output_page"]
            print(pred)
        except Exception as e:
            print(e)

        pred, output_page = "Not_a_plant", "Not_a_plant.html" 
        return render_template(output_page, pred_output = pred, user_image = file_path)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,port=8080) 
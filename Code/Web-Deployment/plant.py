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
        
        file_path = os.path.join('static/upload/test', filename)
        with open(file_path, 'rb') as f:
            file_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Send file name to client
        try:
            result = client.predict({"image": file_data}
                                    , api_name="/predict")
            print(result)
        except Exception as e:
            print("Error: ", e)

        pred, output_page = "Not_a_plant", "Not_a_plant.html" 
        return render_template(output_page, pred_output = pred, user_image = file_path)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,port=8080) 
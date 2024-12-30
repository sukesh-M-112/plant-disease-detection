import os
from flask import Flask, render_template, request, redirect
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define file paths
DISEASE_INFO_PATH = os.path.join(BASE_DIR, 'disease_info.csv')
SUPPLEMENT_INFO_PATH = os.path.join(BASE_DIR, 'supplement_info.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'plant_disease_model_1_latest.pt')

# Load CSV files
try:
    disease_info = pd.read_csv(DISEASE_INFO_PATH, encoding='cp1252')
    supplement_info = pd.read_csv(SUPPLEMENT_INFO_PATH, encoding='cp1252')
    logger.info("CSV files loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"CSV file not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading CSV files: {e}")
    raise

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN.CNN(39)  # Adjust the class count if necessary
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Prediction function
def prediction(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image).unsqueeze(0).to(device)
        output = model(input_data)
        output = output.cpu().detach().numpy()
        index = np.argmax(output)
        return index
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        # Check if an image file is uploaded
        if 'image' not in request.files or request.files['image'].filename == '':
            return render_template('error.html', error_message='No file uploaded.')

        image = request.files['image']
        filename = image.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            # Save the uploaded image
            image.save(file_path)
            logger.info(f"File saved at {file_path}")

            # Perform prediction
            pred = prediction(file_path)

            # Fetch disease and supplement information
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]

            # Render results page
            return render_template(
                'submit.html',
                title=title,
                desc=description,
                prevent=prevent,
                image_url=image_url,
                pred=pred,
                sname=supplement_name,
                simage=supplement_image_url,
                buy_link=supplement_buy_link
            )
        except Exception as e:
            logger.error(f"Error during submission: {e}")
            return render_template('error.html', error_message=str(e))

    return redirect('/')

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

if __name__ == '__main__':
    app.run(debug=True)


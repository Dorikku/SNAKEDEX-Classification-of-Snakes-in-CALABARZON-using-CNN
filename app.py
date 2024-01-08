# app_with_yolo.py
from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
from database import snakes


app = Flask(__name__)

def predict_disease(image_path):
    # Load the YOLOv5 model
    model = YOLO('best.pt')

    # Read and resize the input image using Pillow (PIL)
    
    with Image.open(image_path) as img:
        img = img.resize((255, 255))
    
        # Convert image data to a list

    img = image_path
    # Perform prediction on the image
    results = model(img, save=True, project="static", exist_ok=True)

    # Extract relevant information from the prediction
    # names_dict = results[0].names
    probs = results[0].boxes.data[:, 4:6].tolist()
    snake_index = int(probs[0][1])
    # prediction = names_dict[probs[0][1]]
    float_number = probs[0][0] * 100
    confidence = f"{float_number:.2f}"

    return snake_index, confidence 


@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']

    # Save the file temporarily (optional)
    image_path = 'temp_image.jpg'
    file.save(image_path)
    # Pass the image to the YOLOv5 model for prediction
    prediction, confidence = predict_disease(image_path)
    prediction_path = "static/predict/" + image_path
    pokisnakes_path = "static/pokisnakes/" + str(prediction) + '.png'
    snake_data = snakes[prediction]
    # Render the template with the prediction result
    return render_template('prediction.html', prediction=snake_data, confidence=confidence, predict_path=prediction_path, pokisnakes=pokisnakes_path)

if __name__ == '__main__':
    app.run(debug=True)


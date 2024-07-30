import os
import logging
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure folders and allowed extensions
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'outputs')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model
model_path = os.getenv('MODEL_PATH', './best (1).pt')  # Update with your actual model path
model = YOLO(model_path)

vegetable_properties = {
    0: {"name": 'brinjal'},
    1: {"name": 'capsicum'},
    2: {"name": 'cauliflower'},
    3: {"name": 'corn'},
    4: {"name": 'onion'},
    5: {"name": 'potato'},
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logging.warning("No file part in request")
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        logging.warning("No file selected")
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure the file is overwritten if it exists
        file.save(filepath)
        
        selected_vegetables = request.form.getlist('vegetables')
        return redirect(url_for('process_file', filename=filename, selected_vegetables=','.join(selected_vegetables)))
    
    logging.warning("Invalid file type")
    return redirect(request.url)

@app.route('/process/<filename>/<selected_vegetables>', methods=['GET'])
def process_file(filename, selected_vegetables):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        logging.error(f"File {filename} not found")
        return "File not found", 404

    results = model.predict(source=filepath)

    # Save the image with bounding boxes
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    results[0].save(output_path)
    
    selected_vegetables = selected_vegetables.split(',')

    # Process prediction results
    detected_info = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # Accessing coordinates
            conf = box.conf  # Confidence score
            cls = int(box.cls)  # Class label, converted to int

            if cls in vegetable_properties:
                detected_info.append({
                    "name": vegetable_properties[cls]["name"],
                    "count": 1,
                })

    # Aggregate the results by vegetable name
    aggregated_info = {}
    for info in detected_info:
        name = info["name"]
        if name in aggregated_info:
            aggregated_info[name]["count"] += 1
        else:
            aggregated_info[name] = info

    # Ensure all selected vegetables are included in the results
    for veg in selected_vegetables:
        if veg not in aggregated_info:
            aggregated_info[veg] = {"name": veg, "count": 0}

    return render_template('results.html', filename=filename, detected_info=aggregated_info)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    # Create necessary directories if they do not exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    # Run the app
    app.run(debug=False, host='0.0.0.0')  # For deployment, set debug=False and use host='0.0.0.0'

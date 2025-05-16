# 🐶🐱 Dog vs Cat Image Classification Web App

A deep learning-based image classification web application built with Flask and TensorFlow/Keras that predicts whether an uploaded image is of a **dog** or a **cat**.

---

## 🧠 Project Overview

- **Model Training**: The model is trained using TensorFlow/Keras in a Jupyter Notebook (`app.ipynb`).
- **Deployment**: The trained model (`Model.h5`) is loaded in a Flask application (`app.py`) to serve predictions.
- **Input**: Image file (JPEG/PNG)
- **Output**: Either `"Dog"` or `"Cat"`

---

## 🛠️ Tech Stack

- Python 3
- TensorFlow / Keras
- Flask
- NumPy
- Pillow (PIL)
- HTML (Jinja2 templates)

---

## 📁 Project Structure

├── [app.ipynb](https://github.com/BandaJithendra/Dog-vs-Cat-Image_Classification/blob/main/app.ipynb) # Notebook used for model training <br/>
├── [app.py](https://github.com/BandaJithendra/Dog-vs-Cat-Image_Classification/blob/main/app.py) # Flask app for serving predictions <br/>
├── [Model.h5](https://github.com/BandaJithendra/Dog-vs-Cat-Image_Classification/blob/main/Model.h5) # Trained CNN model <br/>
├── templates/ <br/>
│ ├── Home.html # Upload image page <br/>
│ └── output.html # Display prediction result <br/>
├── static/ # Contains UI Images <br/>
├── Training data/ # Contains training dataset <br/>
├── Test data/ # Contains test dataset <br/>
└── requirements.txt # Python dependencies <br/>


---

## 🚀 Getting Started


```bash
### 1. Clone the Repository
git clone https://github.com/BandaJithendra/Dog-vs-Cat-Image_Classification.git
cd Dog-vs-Cat-Image_Classification

# 2. Create and Activate a Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Train the Model (Optional)
# Open app.ipynb and run the notebook to (re)train and save the model as Model.h5.

# 5. Run the Web App
python app.py

# 6. Use the App
# Go to http://localhost:5000/ in your browser and upload an image to classify.
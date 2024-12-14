# Signature Verification System

## Overview
The **Signature Verification System** is a machine learning-based application designed to authenticate handwritten signatures. This system uses advanced deep learning techniques such as **Convolutional Neural Networks (CNNs)**, **Recurrent Neural Networks (RNNs)**, and **Bidirectional Neural Networks (BiRNNs)** to analyze and verify signatures with high accuracy. 

## Features
- Upload signature images for verification.
- Real-time signature processing and analysis.
- Utilizes hybrid neural networks (CNN + RNN + BiRNN) for robust feature extraction and sequence analysis.
- Comprehensive user-friendly web interface for accessibility.

## Project Architecture
1. **Data Preprocessing**:
   - Resize, normalize, and augment the input signature images.
   - Extract key features for training.

2. **Model Architecture**:
   - **CNNs**: Extract spatial features from signature images.
   - **RNNs (LSTM)**: Analyze temporal or sequential patterns in signatures.
   - **BiRNNs**: Enhance feature learning by processing sequences in both forward and backward directions.

3. **Training**:
   - Train the model using labeled datasets of genuine and forged signatures.
   - Optimize using loss functions like binary cross-entropy and metrics like accuracy and F1-score.

4. **Deployment**:
   - Integrate the trained model into a Django web application.
   - Provide a web interface for signature upload and verification results.

## Technologies Used
### Backend
- **Python 3.9**
- **Django 4.0**
- **TensorFlow** and **Keras** for model training and inference

### Frontend
- HTML5, CSS3, JavaScript for the web interface

### Libraries and Frameworks
- **NumPy**, **Pandas** for data handling
- **OpenCV** for image processing
- **Matplotlib** for data visualization
- **SQLite** for database management in Django

## Installation
### Prerequisites
- Python 3.9+
- Virtual Environment (optional but recommended)
- TensorFlow and Keras installed

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/signature-verification-system.git
   cd signature-verification-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run database migrations:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

4. Start the server:
   ```bash
   python manage.py runserver
   ```

5. Access the application at `http://127.0.0.1:8000/`.

## Usage
1. Register or log in to the application.
2. Upload a signature image for verification.
3. View the verification result: `Genuine` or `Forged`.

## Dataset
- Publicly available signature datasets such as **CEDAR** or **GPDS** were used for training and testing.
- Ensure data is preprocessed (resizing, normalization, and augmentation).

## Model Workflow
1. **Input**: Signature image uploaded by the user.
2. **Preprocessing**: Resize and normalize the image.
3. **Prediction**: The trained model predicts whether the signature is genuine or forged.
4. **Output**: Display result with confidence score.

## File Structure
```
project/
├── signature_verification/         # Django app for signature verification
│   ├── models.py                  # Model definition
│   ├── views.py                   # Core logic for verification
│   ├── templates/                 # HTML templates
├── static/                        # CSS, JavaScript, and image assets
├── signature_model/               # Trained model files
├── media/                         # Uploaded signature images
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
```

## Future Enhancements
- Enhance model accuracy by exploring transformer-based architectures.
- Expand dataset for more diverse signature styles.
- Introduce real-time signature drawing and verification.
- Optimize for deployment on cloud platforms like AWS or Google Cloud.

## Contributors
**Rahul**

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Verifying!** 🎉

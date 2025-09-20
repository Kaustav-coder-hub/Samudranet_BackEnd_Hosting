# SIH-OCEAN Backend

This is the backend service for the SIH-OCEAN project. It provides APIs for:
- Image classification
- Otolith classification
- eDNA classification

The backend is built with **Flask** and uses pre-trained machine learning models.

## Project Structure

```
backend/
├── app.py                      # Main Flask app
├── image_classification.py     # Image classification logic
├── otolith_classification.py   # Otolith classification logic
├── edna_classification_clean.py# eDNA classification logic
├── fish_vectors.pkl            # Model/data file
├── species-edna_vectors.pkl    # Model/data file
├── requirements.txt            # Python dependencies
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd backend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server:**
   ```bash
   python app.py
   ```
   The server will start on `http://localhost:5000`.

## API Endpoints

- `POST /classify/predict` — Image classification
- `POST /otolith/predict` — Otolith classification
- `POST /edna/predict` — eDNA classification

## Deployment

To deploy on Render or similar platforms, ensure:
- The app runs on `host='0.0.0.0'` and uses the `PORT` environment variable.
- All dependencies are listed in `requirements.txt`.

---

## requirements.txt Review

Please ensure your `requirements.txt` includes at least:

```
flask
numpy
pandas
scikit-learn
pillow
```

**Depending on your model code, you may also need:**
- `torch` or `tensorflow` (if using PyTorch or TensorFlow)
- `joblib` (for loading `.pkl` files)
- `gunicorn` (for production deployment on Render)

**Example (edit as needed):**
```
flask
numpy
pandas
scikit-learn
pillow
joblib
gunicorn

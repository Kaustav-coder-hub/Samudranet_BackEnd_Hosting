import os
import torch
import clip
from PIL import Image
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import warnings

image_classification_bp = Blueprint('image_classification', __name__)
# Canonical mapping for the 5 supported species to align backend↔frontend
_CANON_MAP = {
    'setipinna tenuifilis': {
        'name': 'Setipinna tenuifilis',
        'id': 'setipinna_tenuifilis'
    },
    'euterpina acutifrons': {
        'name': 'Euterpina acutifrons',
        'id': 'euterpina_acutifrons'
    },
    'muraenesox cinereus': {
        'name': 'Muraenesox cinereus',
        'id': 'muraenesox_cinereus'
    },
    'nototeredo edax': {
        'name': 'Nototeredo edax',
        'id': 'nototeredo_edax'
    },
    'fiona': {
        'name': 'Fiona (genus)',
        'id': 'fiona_genus'
    },
    'fiona genus': {
        'name': 'Fiona (genus)',
        'id': 'fiona_genus'
    },
    'fiona (genus)': {
        'name': 'Fiona (genus)',
        'id': 'fiona_genus'
    },
}

def _canonicalize(species_str: str):
    key = species_str.strip().lower().replace('_', ' ')
    # remove extra parentheses for matching
    key_simple = key.replace('(', ' ').replace(')', ' ').replace('  ', ' ').strip()
    return _CANON_MAP.get(key) or _CANON_MAP.get(key_simple)


# Load database
with open("fish_vectors.pkl", "rb") as f:
    all_data = pickle.load(f)

# Prepare data
X = np.array([item['vector'] for item in all_data], dtype=np.float32)
y = np.array([item['species'] for item in all_data])
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train classifier on CLIP vectors (with scaling for stability)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf = LogisticRegression(max_iter=2000, n_jobs=None, solver='lbfgs', multi_class='auto')
clf.fit(X_scaled, y_encoded)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
model.eval()

# Classify a new image
def classify_image(image_path):
    # Robust image open and preprocessing
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        image_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        # Avoid zero-division; clamp norm
        norm = embedding.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        embedding = embedding / norm
        vector_t = embedding.float().cpu()

    vector = vector_t.numpy().astype(np.float32)
    # Scale like training
    vector_scaled = scaler.transform(vector)

    # Predict probabilities
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = clf.predict_proba(vector_scaled)[0]

    # Fix any numerical issues
    if not np.isfinite(proba).all():
        proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize to sum to 1
    total = proba.sum()
    if total <= 0:
        # fallback to uniform tiny probs to avoid NaN
        proba = np.full_like(proba, 1.0 / len(proba))
    else:
        proba = proba / total

    pred_idx = int(np.argmax(proba))
    species_raw = le.inverse_transform([pred_idx])[0]
    canon = _canonicalize(species_raw) or {'name': species_raw, 'id': None}
    confidence = float(proba[pred_idx])

    # Top-5 predictions
    top_indices = np.argsort(proba)[::-1][:5]
    top5 = [
        {
            'species': (_canonicalize(le.inverse_transform([int(i)])[0]) or {'name': le.inverse_transform([int(i)])[0]}).get('name'),
            'confidence': float(proba[int(i)])
        }
        for i in top_indices
    ]

    return canon['name'], confidence, vector, proba, top5, canon.get('id')



@image_classification_bp.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp:
        file.save(temp.name)
        temp_path = temp.name
    try:
        result, confidence, test_vector, proba, top5, species_id = classify_image(temp_path)
        # Build focus set for visualization
        mask = (y == result)
        X_known = X[mask]
        # If not enough samples for PCA/TSNE, augment with nearest neighbors across all species
        if X_known.shape[0] < 2:
            # compute cosine similarity to pick neighbors
            tv = test_vector.reshape(1, -1)
            # normalize for cosine sim
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            tv_norm = tv / (np.linalg.norm(tv, axis=1, keepdims=True) + 1e-8)
            sims = (X_norm @ tv_norm.T).reshape(-1)
            # pick top K neighbors
            k = int(min(50, max(5, len(sims) // 10)))
            nn_idx = np.argsort(sims)[::-1][:k]
            X_focus = np.vstack([X[nn_idx], test_vector])
            y_focus = np.append(y[nn_idx], 'Test Image')
        else:
            X_focus = np.vstack([X_known, test_vector])
            y_focus = np.append(y[mask], 'Test Image')

        # PCA with guards
        if X_focus.shape[0] >= 2:
            pca = PCA(n_components=2)
            try:
                X_pca = pca.fit_transform(X_focus)
            except Exception:
                # fallback to first two dims
                X_pca = X_focus[:, :2] if X_focus.shape[1] >= 2 else np.hstack([X_focus, X_focus])
        else:
            X_pca = np.zeros((X_focus.shape[0], 2), dtype=float)
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.scatter(X_pca[:-1,0], X_pca[:-1,1], label=result, alpha=0.6)
        plt.scatter(X_pca[-1,0], X_pca[-1,1], c='red', marker='*', s=200, label='Test Image')
        plt.title('PCA Visualization')
        plt.legend()
        # t-SNE with guards
        if X_focus.shape[0] >= 3:
            perplexity = max(2, min(30, X_focus.shape[0] - 1))
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                X_tsne = tsne.fit_transform(X_focus)
            except Exception:
                X_tsne = X_pca.copy()
        else:
            X_tsne = X_pca.copy()
        plt.subplot(1,2,2)
        plt.scatter(X_tsne[:-1,0], X_tsne[:-1,1], label=result, alpha=0.6)
        plt.scatter(X_tsne[-1,0], X_tsne[-1,1], c='red', marker='*', s=200, label='Test Image')
        plt.title('t-SNE Visualization')
        plt.legend()
        plt.tight_layout()
        # Save plot to temp file
        plot_path = tempfile.mktemp(suffix='.png')
        plt.savefig(plot_path)
        plt.close()
        # Read plot as base64
        import base64
        with open(plot_path, 'rb') as img_f:
            plot_b64 = base64.b64encode(img_f.read()).decode('utf-8')
        os.remove(plot_path)
        # Clean up temp image
        os.remove(temp_path)
        # Prepare numeric embeddings for frontend plotting if needed
        pca_points = {
            'known': X_pca[:-1].tolist(),
            'test': X_pca[-1].tolist()
        }
        tsne_points = {
            'known': X_tsne[:-1].tolist(),
            'test': X_tsne[-1].tolist()
        }
        # Optional map points if coordinates exist in dataset entries
        map_points = []
        for item in all_data:
            if item.get('species') == result and 'lat' in item and 'lon' in item:
                map_points.append({'lat': item['lat'], 'lon': item['lon']})
        return jsonify({
            'species': result,
            'confidence': float(confidence),
            'top5': top5,
            'speciesId': species_id,
            'visualization': plot_b64,
            'pca': pca_points,
            'tsne': tsne_points,
            'mapPoints': map_points
        })
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500



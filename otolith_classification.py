import os
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import base64
import tempfile
import io
import json
from PIL import Image
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

otolith_classification_bp = Blueprint('otolith_classification', __name__)

# Canonical mapping for the supported species to align backend↔frontend
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
    }
}

def _canonicalize(species_str: str):
    key = species_str.strip().lower().replace('_', ' ')
    # remove extra parentheses for matching
    key_simple = key.replace('(', ' ').replace(')', ' ').replace('  ', ' ').strip()
    return _CANON_MAP.get(key) or _CANON_MAP.get(key_simple)

def load_image(image_path):
    """Load and preprocess an otolith image"""
    img = cv2.imread(image_path)
    if img is None:
        # Try opening with PIL if OpenCV fails
        with Image.open(image_path) as pil_img:
            pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
            img = img[:, :, ::-1]  # Convert RGB to BGR for OpenCV
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    return img

def segment_otolith(img):
    """Segment the otolith from the background"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu's thresholding
    _, mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def generate_2d_visualization(img, mask):
    """Generate 2D visualization of the otolith segmentation with higher resolution for zooming"""
    # Create higher resolution image for better zooming
    plt.figure(figsize=(12, 6), dpi=200)
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('2D Segmentation Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_3d_visualization(mask):
    """Generate 3D visualization of the otolith surface from multiple angles"""
    # Apply Gaussian filter for smoothing
    mask_smooth = cv2.GaussianBlur(mask.astype(float), (15, 15), 0)
    
    # Create multiple 3D views from different angles for a rotation effect
    angles = [
        (30, 0),    # Front view
        (30, 45),   # Front-right view
        (30, 90),   # Right view
        (30, 135),  # Back-right view
        (30, 180),  # Back view
        (30, 225),  # Back-left view
        (30, 270),  # Left view
        (30, 315)   # Front-left view
    ]
    
    views = []
    for elev, azim in angles:
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create 3D surface plot with enhanced visual properties
        X, Y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
        Z = mask_smooth
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                              linewidth=0, antialiased=True,
                              alpha=0.8)
        
        # Add a color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Better axis labeling and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Intensity')
        
        # Set specific viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Add to views list
        views.append(base64.b64encode(buf.read()).decode('utf-8'))
    
    return views

def compute_confidence_score(mask):
    """Calculate confidence score based on mask properties"""
    # Basic confidence: proportion of pixels classified as otolith
    basic_score = np.sum(mask) / mask.size
    
    # Add additional confidence metrics
    # 1. Edge coherence (using Sobel filter)
    sobelx = cv2.Sobel(mask.astype(float), cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(mask.astype(float), cv2.CV_64F, 0, 1, ksize=3)
    edge_coherence = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    # 2. Shape compactness
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
    else:
        compactness = 0
    
    # Combine metrics for final confidence score
    confidence = 0.5 * basic_score + 0.3 * edge_coherence + 0.2 * compactness
    # Normalize to [0, 1]
    confidence = min(max(confidence, 0), 1)
    
    return confidence

def generate_confidence_visualization(score):
    """Generate visualization of confidence score"""
    plt.figure(figsize=(4, 5))
    plt.bar(['Otolith Confidence'], [score], color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel('Confidence Score')
    plt.title(f'Otolith Classification Confidence: {score:.2f}')
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def classify_otolith(image_path):
    """Process otolith image and generate classification results"""
    try:
        # Load and process image
        img = load_image(image_path)
        mask = segment_otolith(img)
        
        # Calculate confidence score
        confidence = compute_confidence_score(mask)
        
        # Generate visualizations
        viz_2d = generate_2d_visualization(img, mask)
        viz_3d_views = generate_3d_visualization(mask)
        viz_conf = generate_confidence_visualization(confidence)
        
        # For demo purposes, hardcode species as Setipinna tenuifilis
        # In a real implementation, you would use a trained model here
        species = "Setipinna tenuifilis"
        species_id = "setipinna_tenuifilis"
        
        # Map coordinates for Setipinna tenuifilis (example location in Indian Ocean)
        # These would come from your dataset in a real implementation
        map_points = [
            {'lat': 12.5, 'lon': 80.2},
            {'lat': 12.6, 'lon': 80.3},
            {'lat': 12.4, 'lon': 80.1}
        ]
        
        return {
            'species': species,
            'speciesId': species_id,
            'confidence': float(confidence),
            'visualization2d': viz_2d,
            'visualization3d': viz_3d_views[0],  # Main view (front)
            'visualization3dViews': viz_3d_views,  # All rotation views
            'confidenceViz': viz_conf,
            'mapPoints': map_points
        }
    except Exception as e:
        raise Exception(f"Error in otolith classification: {str(e)}")

@otolith_classification_bp.route('/predict', methods=['POST'])
def predict_otolith():
    """Endpoint for otolith classification"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save uploaded image to temporary file
    filename = secure_filename(file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp:
        file.save(temp.name)
        temp_path = temp.name
    
    try:
        # Process the otolith image
        result = classify_otolith(temp_path)
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        # Return the classification results
        return jsonify(result)
    
    except Exception as e:
        # Clean up the temporary file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({'error': str(e)}), 500

"""
eDNA Classification Module - Professional Implementation
Clean backend for species prediction with advanced 3D DNA visualization
Based on KNN classification with optimized data structures for frontend
"""

import pickle
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io
import base64
import json
from flask import Blueprint, request, jsonify

# Create Flask Blueprint
edna_classification_bp = Blueprint('edna_classification', __name__)

# Global variables
vectors_data = []
knn_model = None
scaler = None
X_train = []
y_train = []

def load_vectors():
    """Load eDNA vectors from pickle file."""
    global vectors_data
    try:
        with open('species-edna_vectors.pkl', 'rb') as f:
            vectors_data = pickle.load(f)
        print(f"✅ Loaded {len(vectors_data)} eDNA vectors from species-edna_vectors.pkl")
        return True
    except Exception as e:
        print(f"❌ Error loading vectors: {e}")
        return False

def extract_features(gc_content, dna_length, sequence=""):
    """Extract features from DNA sequence similar to training data."""
    # Clean sequence
    sequence = re.sub(r'[^ATGC]', '', sequence.upper()) if sequence else ""
    
    features = []
    
    # 1. GC Content
    features.append(float(gc_content))
    
    # 2. DNA Length  
    features.append(float(dna_length))
    
    # 3. Basic nucleotide frequencies
    if len(sequence) > 0:
        a_freq = sequence.count('A') / len(sequence)
        t_freq = sequence.count('T') / len(sequence)
        g_freq = sequence.count('G') / len(sequence)
        c_freq = sequence.count('C') / len(sequence)
    else:
        a_freq, t_freq, g_freq, c_freq = 0, 0, 0, 0
    
    features.extend([a_freq, t_freq, g_freq, c_freq])
    
    return np.array(features)

def train_model():
    """Train KNN model with loaded vector data."""
    global knn_model, scaler, X_train, y_train
    
    if not vectors_data:
        return False
    
    try:
        # Extract training data
        X = []
        y = []
        
        for entry in vectors_data:
            # Extract numerical features from vector (first 6 features)
            vector = []
            for feature in entry['vector']:
                if isinstance(feature, (int, float)):
                    vector.append(feature)
            
            if len(vector) >= 6:
                X.append(vector[:6])  # Use first 6 features to match input
                y.append(entry['species'])
        
        X_train = np.array(X)
        y_train = np.array(y)
        
        # Train KNN classifier
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        
        # Initialize scaler for PCA visualizations
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        print(f"✅ KNN model trained successfully with {len(X_train)} samples")
        print(f"✅ Found {len(np.unique(y_train))} unique species")
        return True
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def create_dna_3d_data(sequence=None):
    """Create 3D DNA double helix data for frontend visualization."""
    
    # Default sequence if none provided
    if not sequence or len(sequence) < 10:
        sequence = "ATGCTAGCTAGATCGATCGATCGACTGATCGATCGATCGAATTCCGGAATTC"
    
    # Clean and validate sequence
    sequence = re.sub(r'[^ATGC]', '', sequence.upper())
    if len(sequence) > 80:  # Limit for performance
        sequence = sequence[:80]
    
    # Base pair mappings
    base_pairs = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    
    # DNA helix parameters (B-form DNA)
    n_points = len(sequence)
    turns = n_points / 10.5  # ~10.5 base pairs per turn
    t = np.linspace(0, 2 * np.pi * turns, n_points)
    radius = 10
    pitch = 34  # Angstroms per turn
    
    # Calculate helix coordinates
    x1 = radius * np.cos(t)
    y1 = radius * np.sin(t)
    z1 = pitch * t / (2 * np.pi)
    
    # Second strand (180° offset)
    x2 = radius * np.cos(t + np.pi)
    y2 = radius * np.sin(t + np.pi)
    z2 = z1
    
    # Professional color scheme for nucleotides
    base_colors = {
        'A': '#FF4444',  # Red - Adenine
        'T': '#4444FF',  # Blue - Thymine  
        'G': '#44AA44',  # Green - Guanine
        'C': '#FFB84D'   # Orange - Cytosine
    }
    
    # Create data structure for frontend
    dna_data = {
        'sequence': sequence,
        'length': len(sequence),
        'gcContent': ((sequence.count('G') + sequence.count('C'))/len(sequence)*100),
        'helicalTurns': len(sequence)/10.5,
        'backbone1': {
            'x': x1.tolist(),
            'y': y1.tolist(),
            'z': z1.tolist()
        },
        'backbone2': {
            'x': x2.tolist(),
            'y': y2.tolist(),
            'z': z2.tolist()
        },
        'nucleotides': [],
        'basePairs': [],
        'baseColors': base_colors
    }
    
    # Add nucleotides and base pairs
    for i, base in enumerate(sequence):
        complement = base_pairs[base]
        
        # First strand nucleotide
        dna_data['nucleotides'].append({
            'base': base,
            'position': i + 1,
            'strand': 1,
            'x': float(x1[i]),
            'y': float(y1[i]),
            'z': float(z1[i]),
            'color': base_colors[base]
        })
        
        # Complementary strand nucleotide
        dna_data['nucleotides'].append({
            'base': complement,
            'position': i + 1,
            'strand': 2,
            'x': float(x2[i]),
            'y': float(y2[i]),
            'z': float(z2[i]),
            'color': base_colors[complement]
        })
        
        # Hydrogen bonds between base pairs
        dna_data['basePairs'].append({
            'position': i + 1,
            'base1': base,
            'base2': complement,
            'x1': float(x1[i]),
            'y1': float(y1[i]),
            'z1': float(z1[i]),
            'x2': float(x2[i]),
            'y2': float(y2[i]),
            'z2': float(z2[i])
        })
    
    return dna_data

def create_analysis_viz(gc_content, dna_length):
    """Create comprehensive DNA analysis visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('eDNA Sequence Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Extract reference data for comparison
    if vectors_data:
        ref_gc = []
        ref_length = []
        
        for entry in vectors_data:
            try:
                if 'gc_content' in entry:
                    ref_gc.append(entry['gc_content'])
                elif len(entry['vector']) > 0:
                    ref_gc.append(entry['vector'][0])
                    
                if 'dna_length' in entry:
                    ref_length.append(entry['dna_length'])
                elif len(entry['vector']) > 1:
                    ref_length.append(entry['vector'][1])
            except:
                continue
    else:
        ref_gc = [gc_content]
        ref_length = [dna_length]
    
    # 1. GC Content Distribution
    axes[0,0].hist(ref_gc, bins=25, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1)
    axes[0,0].axvline(gc_content, color='red', linestyle='--', linewidth=3, 
                     label=f'Input: {gc_content:.1f}%')
    axes[0,0].set_title('GC Content Distribution', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('GC Content (%)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. DNA Length Distribution  
    axes[0,1].hist(ref_length, bins=25, alpha=0.7, color='lightgreen', edgecolor='darkgreen', linewidth=1)
    axes[0,1].axvline(dna_length, color='red', linestyle='--', linewidth=3,
                     label=f'Input: {dna_length:.0f} bp')
    axes[0,1].set_title('DNA Length Distribution', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Length (base pairs)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Feature Comparison Radar Chart (simplified as bar chart)
    if len(ref_gc) > 0 and len(ref_length) > 0:
        features = ['GC Content', 'DNA Length', 'AT Content']
        input_values = [gc_content/100, dna_length/max(ref_length), (100-gc_content)/100]
        ref_means = [np.mean(ref_gc)/100, np.mean(ref_length)/max(ref_length), 
                    (100-np.mean(ref_gc))/100]
        
        x = np.arange(len(features))
        width = 0.35
        
        axes[1,0].bar(x - width/2, input_values, width, label='Input Sample', 
                     color='orange', alpha=0.8)
        axes[1,0].bar(x + width/2, ref_means, width, label='Reference Mean', 
                     color='blue', alpha=0.6)
        
        axes[1,0].set_title('Feature Comparison', fontsize=12, fontweight='bold')
        axes[1,0].set_ylabel('Normalized Value')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(features)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. Quality Metrics
    axes[1,1].axis('off')
    
    # Calculate quality metrics
    gc_percentile = np.percentile(ref_gc, 50) if ref_gc else 50
    length_percentile = np.percentile(ref_length, 50) if ref_length else 500
    
    quality_text = f"""
    SEQUENCE QUALITY METRICS
    
    GC Content: {gc_content:.1f}%
       {'Optimal' if 40 <= gc_content <= 60 else 'Suboptimal' if 30 <= gc_content <= 70 else 'Extreme'}
    
    Length: {dna_length:.0f} bp  
       {'Good' if dna_length >= 100 else 'Short' if dna_length >= 50 else 'Too Short'}
    
    Compared to Reference:
       GC: {'Above' if gc_content > gc_percentile else 'Below'} median
       Length: {'Above' if dna_length > length_percentile else 'Below'} median
    
    Classification Confidence:
       Expected: {'High' if 40 <= gc_content <= 60 and dna_length >= 100 else 'Medium' if dna_length >= 50 else 'Low'}
    """
    
    axes[1,1].text(0.05, 0.95, quality_text, transform=axes[1,1].transAxes, 
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_data = buf.getvalue()
    buf.close()
    plt.close(fig)
    
    return base64.b64encode(img_data).decode('utf-8')

def create_similarity_viz(gc_content, dna_length, predicted_species):
    """Create PCA-based similarity visualization."""
    if scaler is None or len(X_train) == 0:
        return None
    
    try:
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique species and assign colors
        unique_species = np.unique(y_train)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_species)))
        
        # Plot each species cluster
        for i, species in enumerate(unique_species):
            mask = (y_train == species)
            alpha = 1.0 if species == predicted_species else 0.7
            size = 80 if species == predicted_species else 60
            edge_color = 'black' if species == predicted_species else 'white'
            edge_width = 2 if species == predicted_species else 1
            
            scatter = ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=species, alpha=alpha, s=size,
                       edgecolors=edge_color, linewidth=edge_width)
        
        # Plot input sample
        input_features = extract_features(gc_content, dna_length)
        input_scaled = scaler.transform([input_features])
        input_pca = pca.transform(input_scaled)
        
        ax.scatter(input_pca[0, 0], input_pca[0, 1], 
                  color='red', marker='*', s=300, 
                  label='Input Sample', edgecolors='black', linewidth=2,
                  zorder=5)
        
        # Add prediction confidence circle
        distances = np.sqrt(np.sum((X_pca - input_pca[0])**2, axis=1))
        nearest_distance = np.min(distances)
        confidence_circle = plt.Circle((input_pca[0, 0], input_pca[0, 1]), 
                                     nearest_distance * 1.5, 
                                     fill=False, color='red', linestyle='--', alpha=0.5)
        ax.add_patch(confidence_circle)
        
        # Formatting
        ax.set_title(f'Species Similarity Analysis (PCA)\nPredicted: {predicted_species}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_data = buf.getvalue()
        buf.close()
        plt.close(fig)
        
        return base64.b64encode(img_data).decode('utf-8')
        
    except Exception as e:
        print(f"Error creating similarity visualization: {e}")
        return None

@edna_classification_bp.route('/predict', methods=['POST'])
def predict_species():
    """Main endpoint for eDNA species prediction."""
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract parameters
        gc_content = float(data.get('gcContent', 0))
        dna_length = float(data.get('dnaLength', 0))
        sequence = data.get('sequence', '').strip()
        
        # Validate inputs
        if gc_content <= 0 or gc_content > 100:
            return jsonify({'error': 'GC content must be between 0 and 100%'}), 400
        
        if dna_length <= 0:
            return jsonify({'error': 'DNA length must be positive'}), 400
        
        # Process sequence if provided
        if sequence:
            sequence = re.sub(r'[^ATGC]', '', sequence.upper())
            if len(sequence) > 10:
                # Recalculate GC content and length from sequence
                gc_count = sequence.count('G') + sequence.count('C')
                gc_content = (gc_count / len(sequence)) * 100
                dna_length = len(sequence)
        
        # Check if model is available
        if knn_model is None:
            return jsonify({'error': 'Classification model not available'}), 500
        
        # Extract features and make prediction
        features = extract_features(gc_content, dna_length, sequence)
        
        # Predict species
        predicted_species = knn_model.predict([features])[0]
        
        # Get prediction probabilities for confidence
        probabilities = knn_model.predict_proba([features])[0]
        confidence = float(np.max(probabilities))
        
        # Get top 3 predictions
        prob_dict = dict(zip(knn_model.classes_, probabilities))
        top_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Generate visualizations
        dna_analysis_viz = create_analysis_viz(gc_content, dna_length)
        similarity_viz = create_similarity_viz(gc_content, dna_length, predicted_species)
        dna_3d_data = create_dna_3d_data(sequence if sequence else None)
        
        # Prepare comprehensive response
        response = {
            'species': predicted_species,
            'confidence': confidence,
            'topPredictions': [{'species': species, 'probability': float(prob)} 
                             for species, prob in top_predictions],
            'dnaAnalysisViz': dna_analysis_viz,
            'similarityViz': similarity_viz, 
            'dna3dData': dna_3d_data,
            'inputData': {
                'gcContent': gc_content,
                'dnaLength': dna_length,
                'sequenceProvided': bool(sequence),
                'sequenceLength': len(sequence) if sequence else 0
            },
            'modelInfo': {
                'algorithm': 'K-Nearest Neighbors',
                'trainingDataSize': len(X_train),
                'numberOfSpecies': len(np.unique(y_train)),
                'features': ['GC Content', 'DNA Length', 'A Frequency', 'T Frequency', 'G Frequency', 'C Frequency']
            }
        }
        
        print(f"✅ eDNA Prediction: {predicted_species} (confidence: {confidence:.3f})")
        return jsonify(response), 200
        
    except ValueError as e:
        print(f"❌ Input validation error: {e}")
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# Initialize module on import
print("🧬 Initializing eDNA Classification Module...")

if load_vectors():
    if train_model():
        print("✅ eDNA module fully initialized and ready!")
    else:
        print("⚠️ eDNA module loaded but model training failed")
else:
    print("❌ eDNA module failed to load vector data")

print(f"📊 Module status: {'Ready' if knn_model is not None else 'Not Ready'}")

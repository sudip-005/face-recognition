import os
import cv2
import numpy as np
import pickle
import json
from deepface import DeepFace
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DeepFaceTrainer:
    def __init__(self, dataset_path="dataset", model_name="ArcFace"):
        """
        Initialize DeepFace trainer
        model_name: 'ArcFace', 'Facenet', 'Facenet512', 'VGG-Face', 'OpenFace', 'DeepFace'
        """
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.embeddings = []
        self.labels = []
        self.label_names = []  # Store actual names
        self.user_info = {}  # Store user metadata
        
        # Create output directories
        self.output_dir = "deepface_model"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\n" + "="*70)
        print(f"🎯 DEEPFACE TRAINER - Using {model_name}")
        print("="*70)
    
    def load_dataset(self):
        """Load and organize dataset from folders"""
        print("\n📂 Scanning dataset folder...")
        
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path {self.dataset_path} does not exist!")
        
        # Get all user folders
        user_folders = [f for f in os.listdir(self.dataset_path) 
                       if os.path.isdir(os.path.join(self.dataset_path, f))]
        
        print(f"📊 Found {len(user_folders)} user folders")
        
        # Collect all image paths with labels
        self.image_paths = []
        self.image_labels = []
        self.label_to_id = {}
        
        for idx, folder in enumerate(user_folders):
            folder_path = os.path.join(self.dataset_path, folder)
            
            # Extract user name from folder (format: ID_Name or just Name)
            if '_' in folder:
                parts = folder.split('_', 1)
                if len(parts) == 2:
                    user_id = parts[0]
                    user_name = parts[1]
                else:
                    user_id = str(idx)
                    user_name = folder
            else:
                user_id = str(idx)
                user_name = folder
            
            self.label_to_id[user_name] = int(user_id)
            self.label_names.append(user_name)
            
            # Get all images in this folder
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(folder_path, file)
                    self.image_paths.append(img_path)
                    self.image_labels.append(user_name)
        
        print(f"📸 Total images found: {len(self.image_paths)}")
        print(f"👥 Total users: {len(self.label_names)}")
        
        # Print distribution
        print("\n📊 Dataset distribution:")
        from collections import Counter
        label_counts = Counter(self.image_labels)
        for name, count in label_counts.items():
            print(f"   • {name}: {count} images")
    
    def extract_embeddings(self):
        """Extract face embeddings using DeepFace"""
        print(f"\n🔄 Extracting embeddings using {self.model_name}...")
        print("   This may take a while depending on dataset size...")
        
        self.embeddings = []
        self.labels = []
        failed_images = []
        
        for i, (img_path, label) in enumerate(zip(self.image_paths, self.image_labels)):
            try:
                # Show progress
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i+1}/{len(self.image_paths)} images")
                
                # Extract embedding using DeepFace
                result = DeepFace.represent(
                    img_path=img_path,
                    model_name=self.model_name,
                    enforce_detection=False,  # Don't fail if face not detected
                    detector_backend='opencv',
                    align=True
                )
                
                if result and len(result) > 0:
                    embedding = np.array(result[0]['embedding'])
                    self.embeddings.append(embedding)
                    self.labels.append(self.label_to_id[label])
                else:
                    failed_images.append(img_path)
                    
            except Exception as e:
                failed_images.append(img_path)
                print(f"   ⚠️ Failed on {os.path.basename(img_path)}: {str(e)[:50]}")
        
        print(f"\n✅ Successfully processed: {len(self.embeddings)} images")
        print(f"⚠️ Failed: {len(failed_images)} images")
        
        if failed_images:
            print("\n📝 Failed images:")
            for img in failed_images[:5]:
                print(f"   • {os.path.basename(img)}")
            if len(failed_images) > 5:
                print(f"   ... and {len(failed_images) - 5} more")
        
        return len(self.embeddings) > 0
    
    def normalize_embeddings(self):
        """Normalize embeddings for better comparison"""
        if len(self.embeddings) == 0:
            return
        
        print("\n🔄 Normalizing embeddings...")
        self.embeddings = np.array(self.embeddings)
        self.labels = np.array(self.labels)
        
        # L2 Normalization
        l2_normalizer = Normalizer('l2')
        self.embeddings = l2_normalizer.fit_transform(self.embeddings)
        print("✅ Embeddings normalized")
    
    def save_model(self):
        """Save the trained model and metadata"""
        print("\n💾 Saving model and metadata...")
        
        # Save embeddings and labels
        model_data = {
            'embeddings': self.embeddings,
            'labels': self.labels,
            'label_names': self.label_names,
            'label_to_id': self.label_to_id,
            'model_name': self.model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_samples': len(self.embeddings),
            'num_classes': len(self.label_names),
            'embedding_shape': self.embeddings[0].shape if len(self.embeddings) > 0 else None
        }
        
        # Save as pickle
        model_path = os.path.join(self.output_dir, f"{self.model_name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save as JSON for metadata (without embeddings)
        metadata = {k: v for k, v in model_data.items() if k != 'embeddings'}
        metadata_path = os.path.join(self.output_dir, f"{self.model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            # Convert numpy arrays to lists for JSON
            metadata['labels'] = metadata['labels'].tolist() if isinstance(metadata['labels'], np.ndarray) else metadata['labels']
            json.dump(metadata, f, indent=4)
        
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Metadata saved to: {metadata_path}")
        
        return model_path
    
    def calculate_statistics(self):
        """Calculate and display training statistics"""
        print("\n" + "="*70)
        print("📊 TRAINING STATISTICS")
        print("="*70)
        
        print(f"🤖 Model: {self.model_name}")
        print(f"📸 Total samples: {len(self.embeddings)}")
        print(f"👥 Total classes: {len(self.label_names)}")
        print(f"📏 Embedding dimension: {self.embeddings[0].shape[0] if len(self.embeddings) > 0 else 0}")
        
        # Per-class statistics
        print("\n📋 Per-class samples:")
        from collections import Counter
        label_counts = Counter(self.labels)
        
        # Create reverse mapping
        id_to_name = {v: k for k, v in self.label_to_id.items()}
        
        for label_id, count in sorted(label_counts.items()):
            name = id_to_name.get(label_id, f"Class_{label_id}")
            percentage = (count / len(self.embeddings)) * 100
            print(f"   • {name}: {count} samples ({percentage:.1f}%)")
        
        # Calculate average embeddings per class
        print("\n📊 Embedding statistics:")
        print(f"   • Mean magnitude: {np.mean(np.linalg.norm(self.embeddings, axis=1)):.3f}")
        print(f"   • Std magnitude: {np.std(np.linalg.norm(self.embeddings, axis=1)):.3f}")
    
    def test_sample_predictions(self, num_tests=5):
        """Test predictions on random samples from training set"""
        if len(self.embeddings) < num_tests:
            num_tests = len(self.embeddings)
        
        print("\n" + "="*70)
        print("🔍 TESTING PREDICTIONS")
        print("="*70)
        
        # Randomly select samples
        indices = np.random.choice(len(self.embeddings), num_tests, replace=False)
        
        id_to_name = {v: k for k, v in self.label_to_id.items()}
        
        for idx in indices:
            test_emb = self.embeddings[idx]
            true_label = self.labels[idx]
            true_name = id_to_name.get(true_label, f"Class_{true_label}")
            
            # Calculate similarities with all embeddings
            similarities = []
            for emb in self.embeddings:
                sim = 1 - cosine(test_emb, emb)
                similarities.append(sim)
            
            # Find top 3 matches
            top_indices = np.argsort(similarities)[-3:][::-1]
            
            print(f"\n📝 Testing sample {idx}:")
            print(f"   True: {true_name}")
            print(f"   Top matches:")
            
            for i, match_idx in enumerate(top_indices):
                match_label = self.labels[match_idx]
                match_name = id_to_name.get(match_label, f"Class_{match_label}")
                match_sim = similarities[match_idx]
                is_correct = "✓" if match_label == true_label else "✗"
                print(f"      {i+1}. {match_name}: {match_sim:.3f} {is_correct}")
    
    def visualize_embeddings(self):
        """Visualize embeddings using PCA"""
        if len(self.embeddings) < 2:
            print("⚠️ Not enough samples for visualization")
            return
        
        try:
            from sklearn.decomposition import PCA
            
            print("\n📊 Visualizing embeddings with PCA...")
            
            # Reduce to 2D for visualization
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(self.embeddings)
            
            # Create reverse mapping
            id_to_name = {v: k for k, v in self.label_to_id.items()}
            
            # Plot
            plt.figure(figsize=(12, 8))
            
            # Different colors for different classes
            unique_labels = np.unique(self.labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = self.labels == label
                name = id_to_name.get(label, f"Class_{label}")
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          c=[color], label=name, alpha=0.6, s=50)
            
            plt.title(f'Face Embeddings Visualization - {self.model_name}')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(self.output_dir, f"{self.model_name}_visualization.png")
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            print(f"✅ Visualization saved to: {plot_path}")
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not visualize: {e}")
    
    def train(self):
        """Main training pipeline"""
        print("\n" + "="*70)
        print("🚀 STARTING DEEPFACE TRAINING")
        print("="*70)
        
        # Step 1: Load dataset
        self.load_dataset()
        
        if len(self.image_paths) == 0:
            print("❌ No images found!")
            return False
        
        # Step 2: Extract embeddings
        success = self.extract_embeddings()
        
        if not success or len(self.embeddings) == 0:
            print("❌ Failed to extract embeddings!")
            return False
        
        # Step 3: Normalize
        self.normalize_embeddings()
        
        # Step 4: Calculate statistics
        self.calculate_statistics()
        
        # Step 5: Save model
        model_path = self.save_model()
        
        # Step 6: Test predictions
        self.test_sample_predictions(min(5, len(self.embeddings)))
        
        # Step 7: Visualize
        self.visualize_embeddings()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"📁 Model saved to: {model_path}")
        
        return True

def main():
    """Main function to run training"""
    print("\n" + "="*70)
    print("🎯 DEEPFACE FACE RECOGNITION TRAINER")
    print("="*70)
    
    # Available models
    models = ['ArcFace', 'Facenet', 'Facenet512', 'VGG-Face', 'OpenFace', 'DeepFace']
    
    print("\n📋 Available models:")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model}")
    
    # Model selection
    try:
        choice = input("\nSelect model (1-6, default=1): ").strip()
        if choice and choice.isdigit() and 1 <= int(choice) <= len(models):
            model_name = models[int(choice) - 1]
        else:
            model_name = 'ArcFace'  # Default
    except:
        model_name = 'ArcFace'
    
    # Dataset path
    dataset_path = input(f"Enter dataset path [default: dataset]: ").strip()
    if not dataset_path:
        dataset_path = "dataset"
    
    # Create trainer and run
    trainer = DeepFaceTrainer(dataset_path=dataset_path, model_name=model_name)
    trainer.train()

if __name__ == "__main__":
    main()
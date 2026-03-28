import cv2
import numpy as np
import os
from datetime import datetime
import json
class FaceDatasetCollector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cam = cv2.VideoCapture(0)
        self.dataset_dir = 'dataset'
        
        # Create main directory
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            print(f"📁 Created main directory: {self.dataset_dir}")
    
    def collect_dataset(self):
        print("\n" + "="*60)
        print("🎯 FACE DATASET COLLECTION SYSTEM")
        print("="*60)
        
        # Get user information
        user_id = input('Enter User ID (numbers only): ').strip()
        name = input('Enter Full Name: ').strip()
        
        # Create user folder
        user_folder = os.path.join(self.dataset_dir, f"{user_id}_{name}")
        
        if os.path.exists(user_folder):
            print(f"⚠️ User folder already exists: {user_folder}")
            choice = input("Choose: [O]verwrite, [A]ppend, [C]ancel: ").lower()
            
            if choice == 'c':
                print("❌ Operation cancelled")
                return
            elif choice == 'o':
                # Delete existing images
                for file in os.listdir(user_folder):
                    os.remove(os.path.join(user_folder, file))
                print("🗑️ Existing images deleted")
            elif choice == 'a':
                # Count existing images
                existing = len([f for f in os.listdir(user_folder) if f.endswith('.jpg')])
                print(f"📊 Found {existing} existing images")
        else:
            os.makedirs(user_folder)
            print(f"📁 Created folder: {user_folder}")
        
        # Save user metadata
        metadata = {
            'id': user_id,
            'name': name,
            'age': age,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_images': 0
        }
        
        metadata_file = os.path.join(user_folder, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print("\n🎥 Instructions:")
        print("   • Look directly at the camera")
        print("   • Move your head slightly for variety")
        print("   • Press 'c' to capture an image")
        print("   • Press 'a' for auto-capture mode")
        print("   • Press 'q' to quit")
        print("   • Target: 100 images for best results")
        
        sample_count = len([f for f in os.listdir(user_folder) if f.endswith('.jpg')])
        auto_mode = False
        
        while True:
            ret, frame = self.cam.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            # Mirror effect
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Process each face
            for (x, y, w, h) in faces:
                # Draw rectangle
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add information
                cv2.putText(display, f"{name}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display, f"ID: {user_id}", (x, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Auto-capture if enabled
                if auto_mode and sample_count < 100:
                    sample_count += 1
                    self.save_face_sample(frame, gray, x, y, w, h, user_folder, user_id, name, sample_count)
            
            # Display information
            cv2.putText(display, f"Images: {sample_count}/100", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, f"Mode: {'AUTO' if auto_mode else 'MANUAL'}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display, "Commands: [c]apture [a]uto [q]uit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow("Face Dataset Collector", display)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Manual capture
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    sample_count += 1
                    self.save_face_sample(frame, gray, x, y, w, h, user_folder, user_id, name, sample_count)
                    print(f"✅ Captured: {sample_count}")
                else:
                    print("❌ No face detected!")
            
            elif key == ord('a'):  # Toggle auto mode
                auto_mode = not auto_mode
                print(f"🔄 Auto mode: {'ON' if auto_mode else 'OFF'}")
            
            elif key == ord('q'):  # Quit
                print("\n👋 Exiting...")
                break
            
            # Stop if we have enough images
            if sample_count >= 100:
                print("\n✅ Target of 100 images reached!")
                break
        
        # Update metadata
        metadata['total_images'] = sample_count
        metadata['completed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.cam.release()
        cv2.destroyAllWindows()
        
        self.show_summary(user_folder, sample_count)
    
    def save_face_sample(self, color_img, gray_img, x, y, w, h, user_folder, user_id, name, count):
        """Save both grayscale and color face samples"""
        # Extract face regions
        face_gray = gray_img[y:y+h, x:x+w]
        face_color = color_img[y:y+h, x:x+w]
        
        # Resize to standard size
        face_gray = cv2.resize(face_gray, (200, 200))
        face_color = cv2.resize(face_color, (200, 200))
        
        # Save images
        gray_filename = os.path.join(user_folder, f"{user_id}_{name}_{count:03d}_gray.jpg")
        color_filename = os.path.join(user_folder, f"{user_id}_{name}_{count:03d}_color.jpg")
        
        cv2.imwrite(gray_filename, face_gray)
        cv2.imwrite(color_filename, face_color)
        
        # Also save with variations for augmentation
        if count % 5 == 0:  # Every 5th image, save variations
            self.save_variations(face_gray, face_color, user_folder, user_id, name, count)
    
    def save_variations(self, face_gray, face_color, user_folder, user_id, name, count):
        """Save variations of the face for better training"""
        # Horizontal flip
        flipped_gray = cv2.flip(face_gray, 1)
        flipped_color = cv2.flip(face_color, 1)
        
        cv2.imwrite(os.path.join(user_folder, f"{user_id}_{name}_{count:03d}_flip_gray.jpg"), flipped_gray)
        cv2.imwrite(os.path.join(user_folder, f"{user_id}_{name}_{count:03d}_flip_color.jpg"), flipped_color)
        
        # Brightness variations
        bright_color = cv2.convertScaleAbs(face_color, alpha=1.2, beta=20)
        dark_color = cv2.convertScaleAbs(face_color, alpha=0.8, beta=-20)
        
        cv2.imwrite(os.path.join(user_folder, f"{user_id}_{name}_{count:03d}_bright.jpg"), bright_color)
        cv2.imwrite(os.path.join(user_folder, f"{user_id}_{name}_{count:03d}_dark.jpg"), dark_color)
    
    def show_summary(self, user_folder, sample_count):
        """Show collection summary"""
        print("\n" + "="*60)
        print("📊 DATASET COLLECTION SUMMARY")
        print("="*60)
        print(f"📁 Location: {user_folder}")
        print(f"🖼️ Total images: {sample_count}")
        
        # List all files
        print("\n📄 Files created:")
        files = sorted(os.listdir(user_folder))
        for file in files[:10]:  # Show first 10
            size = os.path.getsize(os.path.join(user_folder, file)) // 1024
            print(f"   📄 {file} ({size} KB)")
        
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more files")
        
        print(f"\n✅ Dataset collection complete!")
        print(f"📊 Total images saved: {sample_count}")
        print(f"📁 Folder path: {user_folder}")

# Run the collector
if __name__ == "__main__":
    collector = FaceDatasetCollector()
    collector.collect_dataset()

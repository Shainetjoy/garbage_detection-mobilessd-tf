"""
Live Detection Script using EfficientNet
Real-time waste classification from webcam or images
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import json
import argparse
from pathlib import Path
import time

class LiveWasteDetector:
    def __init__(self, model_path, class_info_path=None, device='cuda'):
        """Initialize the detector with trained model"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load class information
        if class_info_path is None:
            class_info_path = Path(model_path).parent / 'class_info.json'
        
        if Path(class_info_path).exists():
            with open(class_info_path, 'r') as f:
                class_info = json.load(f)
            self.classes = class_info['classes']
            self.idx_to_class = {int(k): v for k, v in class_info['idx_to_class'].items()}
        else:
            print("Warning: class_info.json not found. Using default classes.")
            self.classes = ['cardboard', 'clothes', 'glass', 'paper', 'plastic', 'shoes', 'trash']
            self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}
        
        # Load model
        import torchvision
        model = torchvision.models.efficientnet_b0(pretrained=False)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(model.classifier[1].in_features, len(self.classes))
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully!")
        print(f"Classes: {', '.join(self.classes)}")
    
    def predict_image(self, image_path, show=True, save=True):
        """Predict on a single image"""
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return None
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(input_tensor)
            inference_time = time.time() - start_time
            
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            predicted_class = self.idx_to_class[predicted.item()]
            confidence_score = confidence.item()
        
        # Print results
        print("\n" + "=" * 60)
        print("Prediction Results:")
        print("=" * 60)
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence_score:.2%}")
        print(f"Inference Time: {inference_time*1000:.2f} ms")
        print("\nAll Class Probabilities:")
        for idx, cls in self.idx_to_class.items():
            prob = probabilities[idx].item()
            marker = " <--" if idx == predicted.item() else ""
            print(f"  {cls}: {prob:.2%}{marker}")
        print("=" * 60)
        
        # Display image with prediction
        if show:
            img_cv = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            
            # Add text overlay
            text = f"{predicted_class}: {confidence_score:.1%}"
            cv2.putText(img_cv, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_cv, f"Time: {inference_time*1000:.1f}ms", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Waste Classification - Press Q to close', img_cv)
            print("\nPress 'Q' to close the image window...")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or cv2.getWindowProperty('Waste Classification - Press Q to close', cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyAllWindows()
        
        # Save annotated image
        if save:
            output_dir = Path('predictions')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"predicted_{image_path.name}"
            if show:
                cv2.imwrite(str(output_path), img_cv)
            else:
                img_cv = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                cv2.putText(img_cv, f"{predicted_class}: {confidence_score:.1%}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(str(output_path), img_cv)
            print(f"\nAnnotated image saved to: {output_path}")
        
        return {
            'class': predicted_class,
            'confidence': confidence_score,
            'all_probabilities': {self.idx_to_class[i]: probabilities[i].item() 
                                 for i in range(len(self.classes))},
            'inference_time': inference_time
        }
    
    def predict_webcam(self, camera_index=0, conf_threshold=0.5):
        """Live prediction from webcam"""
        print(f"\nStarting webcam feed (Camera {camera_index})...")
        print("Press 'Q' to quit, 'S' to save current frame")
        print(f"Confidence threshold: {conf_threshold:.0%}")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Preprocess
                input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, predicted = torch.max(probabilities, 0)
                    predicted_class = self.idx_to_class[predicted.item()]
                    confidence_score = confidence.item()
                
                # Draw prediction on frame
                if confidence_score >= conf_threshold:
                    color = (0, 255, 0)  # Green
                    text = f"{predicted_class}: {confidence_score:.1%}"
                else:
                    color = (0, 165, 255)  # Orange
                    text = f"{predicted_class}: {confidence_score:.1%} (Low)"
                
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'Q' to quit, 'S' to save", (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show top 3 predictions
                top3_probs, top3_indices = torch.topk(probabilities, min(3, len(self.classes)))
                y_offset = 110
                for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                    cls_name = self.idx_to_class[idx.item()]
                    prob_val = prob.item()
                    cv2.putText(frame, f"{i+1}. {cls_name}: {prob_val:.1%}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 30
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                
                cv2.imshow('Live Waste Classification', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    # Save current frame
                    output_dir = Path('predictions')
                    output_dir.mkdir(exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = output_dir / f"webcam_{timestamp}.jpg"
                    cv2.imwrite(str(output_path), frame)
                    print(f"Frame saved to: {output_path}")
        
        except KeyboardInterrupt:
            print("\nStopping webcam feed...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam feed closed.")

def main():
    parser = argparse.ArgumentParser(description='Live Waste Classification Detection')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to trained model file')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image file')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam for live detection')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold for webcam (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display images')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save prediction results')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Live Waste Classification Detection")
    print("=" * 60)
    
    try:
        detector = LiveWasteDetector(args.model, device=args.device)
        
        if args.webcam:
            detector.predict_webcam(args.camera, args.conf)
        elif args.image:
            detector.predict_image(
                args.image,
                show=not args.no_show,
                save=not args.no_save
            )
        else:
            print("\nError: Please specify one of the following:")
            print("  --image path/to/image.jpg  (for single image)")
            print("  --webcam                    (for live webcam feed)")
            print("\nExample usage:")
            print("  python live_detect.py --image path/to/image.jpg")
            print("  python live_detect.py --webcam")
            parser.print_help()
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease make sure:")
        print("  1. The model has been trained (run: python train_efficientdet.py)")
        print("  2. The model path is correct")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


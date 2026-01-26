import cv2
import os
import time

# --- CONFIGURATION ---
DATA_DIR = './static_models/static_data_images'
os.makedirs(DATA_DIR, exist_ok=True)

# You can expand this list with 'A', 'B', 'NG', '1', '2', etc.
GESTURE_CLASSES = ['E', 'I', 'O', 'U'] 
IMAGES_PER_CLASS = 100
FPS_DELAY = 100 # Adjust delay between frames (ms) if needed

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access camera.")
    exit()

print("Camera initialized. Press 'S' to start or 'Q' to quit.")

# Initial Wait Screen
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "STATIC IMAGE COLLECTOR", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'S' to start or 'Q' to quit", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Data Collection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Main Collection Loop
for gesture in GESTURE_CLASSES:
    gesture_dir = os.path.join(DATA_DIR, gesture)
    os.makedirs(gesture_dir, exist_ok=True)
    
    # Non-overwrite logic: Find how many images already exist
    existing = [f for f in os.listdir(gesture_dir) if f.endswith('.jpg')]
    starting_index = len(existing)
    
    print(f"\nCollecting images for: {gesture}")
    
    # Countdown before starting the class
    for countdown in range(3, 0, -1):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"GET READY: {gesture}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, f"Starting in {countdown}...", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1000)

    # Image capture loop
    counter = 0
    while counter < IMAGES_PER_CLASS:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        # Save frame
        img_name = f"{gesture}_{starting_index + counter}.jpg"
        save_path = os.path.join(gesture_dir, img_name)
        cv2.imwrite(save_path, frame)
        
        # UI Overlay
        cv2.putText(frame, f"Capturing {gesture}: {counter+1}/{IMAGES_PER_CLASS}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("Data Collection", frame)
        
        counter += 1
        
        # Short delay to allow for movement between frames
        if cv2.waitKey(FPS_DELAY) & 0xFF == 27: # Press ESC to skip class
            break

    print(f"✅ Saved {counter} new images to {gesture_dir}")
    time.sleep(1)

cap.release()
cv2.destroyAllWindows()
print("\n🎉 Collection complete!")
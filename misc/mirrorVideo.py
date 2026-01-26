import cv2
import os

def mirror_videos(input_folder, output_folder):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List of filenames to skip (without needing to worry about the extension)
    exclude_files = ['16_1', '16_2']

    # Filter for video files
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    for filename in files:
        # Check if the filename (without extension) is in our exclude list
        file_name_no_ext = os.path.splitext(filename)[0]
        if file_name_no_ext in exclude_files:
            print(f"Skipping: {filename}")
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        cap = cv2.VideoCapture(input_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Mirroring: {filename}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mirrored_frame = cv2.flip(frame, 1)
            out.write(mirrored_frame)

        cap.release()
        out.release()

    print(f"\nDone! Processed videos are in '{output_folder}'.")

# Run the function
mirror_videos('./data_videos/16', '16_flipped')
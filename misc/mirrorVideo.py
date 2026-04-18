import cv2
import os

def mirror_videos(input_folder, output_folder, target_files=None):
    """
    target_files: Can be a single string (filename), 
                  a list of strings, or None for all files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    exclude_files = ['16_1', '16_2']

    # 1. Determine which files to process
    if isinstance(target_files, str):
        files = [target_files]
    elif isinstance(target_files, list):
        files = target_files
    else:
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    for filename in files:
        file_name_no_ext = os.path.splitext(filename)[0]
        
        if file_name_no_ext in exclude_files:
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if not os.path.isfile(input_path):
            print(f"File not found: {filename}")
            continue

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

    print(f"\nDone! Files saved in '{output_folder}'.")

# --- HOW TO RUN FOR 14_0001 TO 14_0011 ---

# Use a list comprehension to generate the names from 1 to 11
# f"{i:04}" ensures the number is 4 digits long (0001, 0002, etc.)
files_to_mirror = [f"Z_{i:04}.mp4" for i in range(1, 12)]

mirror_videos(
    input_folder='./data_videos/ALPHABET+NUMBERS/Z', 
    output_folder='./data_videos/ALPHABET+NUMBERS/Z/Z_flipped', 
    target_files=files_to_mirror
)
import cv2
import os

def retrieve_sub_object(frame, main_object, sub_object, output_path):
    main_x1, main_y1, main_x2, main_y2 = main_object['bbox']
    sub_x1, sub_y1, sub_x2, sub_y2 = sub_object['bbox']

    # Ensure sub-object coordinates are within main object
    sub_x1 = max(0, sub_x1 - main_x1)
    sub_y1 = max(0, sub_y1 - main_y1)
    sub_x2 = min(main_x2 - main_x1, sub_x2 - main_x1)
    sub_y2 = min(main_y2 - main_y1, sub_y2 - main_y1)

    # Crop the sub-object from the main object
    main_object_image = frame[main_y1:main_y2, main_x1:main_x2]
    sub_object_image = main_object_image[sub_y1:sub_y2, sub_x1:sub_x2]

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save the cropped sub-object image
    filename = f"{output_path}/{main_object['object']}_{main_object['id']}_{sub_object['object']}_{sub_object['id']}.jpg"
    cv2.imwrite(filename, sub_object_image)
    print(f"Saved sub-object image: {filename}")

    return filename


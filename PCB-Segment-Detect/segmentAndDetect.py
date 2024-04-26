from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO

component_count = {'ic':0 ,
                   'resistor':0, 
                   'capacitor':0, 
                   'led':0, 
                   'button':0,  
                   'transistor':0}

# Load the model.
model = YOLO('PCB-Segment-Detect/weights/6-classes-kuo-nano.pt')

classes_to_include = ['button', 'capacitor', 'ic', 'led', 'resistor', 'transistor']

def detect_components(img):
    result = model(img, conf=0.20, verbose=False)
    plot = result[0].plot()
    return plot

def split_and_detect(img):

    # Get the dimensions of the image
    width, height, _ = img.shape

    # Segment dimentions
    seg_x = int(0.5 * width)
    seg_y = int(0.5 * height)

    # Calculate the number of segments in each dimension
    num_segments_x = width // seg_x
    num_segments_y = height // seg_y

    # Initialize subplot parameters
    # ~ fig, axs = plt.subplots(num_segments_y, num_segments_x)
    
    stitched_image = np.array([])

    # Iterate through each segment
    for y in range(num_segments_y):
        horizontal_stitch = np.array([])
        for x in range(num_segments_x):
            # Define the bounding box for the segment
            left = x * seg_x
            upper = y * seg_y
            right = min(left + seg_x, width)
            lower = min(upper + seg_y, height)
            # Crop the segment
            segment = img[left:right, upper:lower]
            # ~ segment = img.crop((left, upper, right, lower))
            
            # Run inference on the segment
            result = detect_components(segment)
            
            # Stitch the output of the segment horizontally
            if horizontal_stitch.size == 0:
                horizontal_stitch = np.array(result)
            else:
                horizontal_stitch = np.concatenate((horizontal_stitch, result), axis=0)
        
        # Stitch the horizontal segments vertically
        if stitched_image.size == 0:
            stitched_image = horizontal_stitch
        else:
            stitched_image = np.concatenate((stitched_image, horizontal_stitch), axis=1)
    
    return stitched_image

def main():
    image_path = 'Sample-Images/PCBImage3.jpg'
    
    # Open the image
    image = Image.open(image_path)
    ret = split_and_detect(np.array(image))
    
    # Display the stiched image
    plt.imshow(ret)
    plt.show()
    print(component_count)

if __name__ == "__main__":
    main()

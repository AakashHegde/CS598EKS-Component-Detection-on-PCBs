from PIL import Image
import matplotlib.pyplot as plt

from ultralytics import YOLO

component_count = {'ic':0 ,
                   'resistor':0, 
                   'capacitor':0, 
                   'led':0, 
                   'button':0,  
                   'transistor':0}

# Load the model.
model = YOLO('PCB-Segment-Detect/weights/6-classes-kuo.pt')

classes_to_include = ['button', 'capacitor', 'ic', 'led', 'resistor', 'transistor']

def detect_components(img):
    result = model(img, conf=0.20, verbose=False)
    plot = result[0].plot()
    return plot

def split_and_display(img):
    # Get the dimensions of the image
    width, height = img.size

    # Segment dimentions
    seg_x = int(0.5 * width)
    seg_y = int(0.5 * height)

    # Calculate the number of segments in each dimension
    num_segments_x = width // seg_x
    num_segments_y = height // seg_y

    # Initialize subplot parameters
    fig, axs = plt.subplots(num_segments_y, num_segments_x)

    # Iterate through each segment
    for y in range(num_segments_y):
        for x in range(num_segments_x):
            # Define the bounding box for the segment
            left = x * seg_x
            upper = y * seg_y
            right = min(left + seg_x, width)
            lower = min(upper + seg_y, height)
            # Crop the segment
            segment = img.crop((left, upper, right, lower))
            
            # Run inference on the segment
            result = detect_components(segment)
            
            # Add the result to subplot
            axs[y, x].imshow(result)
            axs[y, x].axis('off')
    
    # Display the subplot
    plt.show()

def main():
    image_path = 'Sample-Images/PCBImage3.jpg'
    
    # Open the image
    image = Image.open(image_path)
    split_and_display(image)
    print(component_count)

if __name__ == "__main__":
    main()

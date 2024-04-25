from PIL import Image
import matplotlib.pyplot as plt

from ultralytics import YOLO

component_count = {'ic':0 ,
                   'resistor':0, 
                   'capacitor':0, 
                   'led':0, 
                   'button':0, 
                   'diode':0, 
                   'inductor':0, 
                   'transistor':0}

# Load the model.
model = YOLO('PCB-Segment-Detect/weights/best.pt')

classes_to_include = ['ic', 'resistor', 'capacitor', 'led', 'button', 'diode', 'inductor', 'transistor']

def split_and_display(image_path):
    # Open the image
    img = Image.open(image_path)

    # Get the dimensions of the image
    width, height = img.size

    # segment dimentions
    seg_x = int(0.33 * width)
    seg_y = int(0.33 * height)

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

            # Display the segment
            axs[y, x].imshow(segment)
            axs[y, x].axis('off')  # Turn off axis labels

            results = model(segment, conf=0.20)

            # Process results list
            for result in results:
                boxes_to_keep = []
                for box in result.boxes:
                    detected_class = model.names[int(box.cls)]
                    if(detected_class in classes_to_include):
                        boxes_to_keep.append(box)
                        component_count[detected_class] += 1
                result.boxes = boxes_to_keep
                # result.show()  # display to screen
    plt.show()



# Example usage
image = 'Sample-Images/PCBImage2.jpg'

split_and_display(image)
print(component_count)

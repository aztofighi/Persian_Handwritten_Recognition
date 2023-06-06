# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import tempfile
 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageOps
from PIL import Image, ImageDraw, ImageFont




class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Load the saved model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')  # Map the model to the CPU

        self.model = resnet50(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 10)  # Assuming you have 10 classes
        self.model.load_state_dict(torch.load("./resnet50_model_cpu.pth"))
        self.model = self.model.to(device)
        self.model.eval()


    def predict(
        self,
        image: Path = Input(description="Input image"),
    ) -> Path:
        
        """Run a single prediction on the model"""
        # Convert the image to tensor and normalize
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')  # Map the model to the CPU

 

        # Convert the image to RGB
        #rgb_image = Image.open(str(image)).convert('RGB')

        image1 = Image.open(str(image)).convert('L')  # Convert to grayscale

        # Convert image to binary
        binary_threshold = 128
        image2 = image1.point(lambda x: 0 if x < binary_threshold else 255)
        
        # Convert image to negative
        image2 = Image.eval(image2, lambda x: 255 - x)

        # Find the outmost non-zero pixel
        width, height = image2.size
        left, top, right, bottom = width, height, 0, 0
        pixels = image2.load()
        for y in range(height):
            for x in range(width):
                if pixels[x, y] > 0:
                    left = min(left, x)
                    top = min(top, y)
                    right = max(right, x)
                    bottom = max(bottom, y)

        # Crop the image to remove zero padding
        image_centered = image2.crop((left, top, right + 1, bottom + 1))
        max_size = max(image_centered.size) 
    
        image3 = image_centered

        # Find the minimum and maximum height and width of the image
        min_size = min(image_centered.size)
        max_size = max(image_centered.size)

        # Calculate the padding size
        padding_size = max_size - min_size

        # Determine the side with the minimum size
        if image_centered.width == min_size:
            padding = (padding_size // 2, 0, padding_size - padding_size // 2, 0)  # Add padding horizontally
        else:
            padding = (0, padding_size // 2, 0, padding_size - padding_size // 2)  # Add padding vertically

        # Apply zero padding to the image
        padded_image = ImageOps.expand(image_centered, padding)

        
        # Resize the image while maintaining aspect ratio
        resize_transform = transforms.Resize((32, 32))
        resized_image = resize_transform(padded_image)

        # Convert the image to RGB
        rgb_image = resized_image.convert('RGB')
        image4 = rgb_image
        
        # Convert the image to tensor and normalize
        to_tensor_transform = transforms.ToTensor()
        image_tensor = to_tensor_transform(rgb_image).unsqueeze(0).to(device)
        image_tensor = (image_tensor - 0.5) / 0.5

        # Forward pass through the model
        with torch.no_grad():
            output = self.model(image_tensor)

         # Define the class labels
        class_labels = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9']


        # Get the predicted class label
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_labels[predicted_idx.item()]

        print("Predicted label:", predicted_label)

        # Stitch the images together
        stitched_image = Image.new('RGB', (rgb_image.width * 8, rgb_image.height))
        stitched_image.paste(image1.convert('RGB').resize((rgb_image.width, rgb_image.height)), (0, 0))
        stitched_image.paste(image2.convert('RGB').resize((rgb_image.width, rgb_image.height)), (rgb_image.width * 1, 0))
        stitched_image.paste(image3.convert('RGB').resize((rgb_image.width, rgb_image.height)), (rgb_image.width * 2, 0))
        stitched_image.paste(image4.convert('RGB').resize((rgb_image.width, rgb_image.height)), (rgb_image.width * 3, 0))

        # Create a new image with white background
        image5 = Image.new("RGB", (200, 100), (255, 255, 255))
        # Create a draw object
        draw = ImageDraw.Draw(image5)
        # Define the font and text
        font = ImageFont.load_default()
        text = f"Class: {predicted_label}"
        # Draw the text on the image
        draw.text((20, 10), text, fill=(0,), font=font)
        stitched_image.paste(image5.convert('RGB'), (rgb_image.width * 4, 0))


        stitched_image = stitched_image.resize((stitched_image.width*2,stitched_image.height*2))

    
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        stitched_image.save(str(out_path))
        return out_path

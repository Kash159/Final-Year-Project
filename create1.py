import torch
import torch.nn as nn  # Import nn module
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


# Define the U-Net model class (include this part or import it if already defined)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.encoder1 = conv_block(1, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = up_conv(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = up_conv(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = up_conv(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = up_conv(128, 64)
        self.decoder1 = conv_block(128, 64)

        self.conv_final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoding
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoding
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return torch.sigmoid(self.conv_final(d1))

# Load the trained model
model = UNet()
model.load_state_dict(torch.load("unet_lung_segmentation2.pth"))
model.eval()  # Set the model to evaluation mode

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Define the transformations for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Define input and output folders
input_folders = [
    "/Users/kashyapkaliyur/Desktop/ML Projects/Final Year Project/Shenzen Dataset/images/Normal",
    "/Users/kashyapkaliyur/Desktop/ML Projects/Final Year Project/Shenzen Dataset/images/TB"
]
output_folder = "/Users/kashyapkaliyur/Desktop/ML Projects/Final Year Project/Predicted_Masks"
os.makedirs(output_folder, exist_ok=True)

# Loop through each folder and process images
for folder in input_folders:
    # Extract folder name (Normal or TB) for organizing output
    folder_name = os.path.basename(folder)
    output_path = os.path.join(output_folder, folder_name)
    os.makedirs(output_path, exist_ok=True)  # Creates subfolder under Predicted_Masks for Normal or TB

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        
        # Load and preprocess the image
        image = Image.open(img_path).convert("L")
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        
        # Predict the mask
        with torch.no_grad():
            predicted_mask = model(image_tensor)
            predicted_mask = predicted_mask.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU

        # Convert to binary mask (0 and 255) for visualization
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255  # Threshold and scale for saving
        
        # Convert array back to an image
        predicted_mask_img = Image.fromarray(predicted_mask.squeeze(), mode="L")
        
        # Save the predicted mask
        mask_output_path = os.path.join(output_path, f"{os.path.splitext(img_name)[0]}_mask.png")
        predicted_mask_img.save(mask_output_path)  # Save the mask image here

        print(f"Processed and saved mask for: {img_name} in {mask_output_path}")

print("All images processed and masks saved.")


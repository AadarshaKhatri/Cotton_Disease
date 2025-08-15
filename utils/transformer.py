from torchvision import transforms

def transformer():
  Image_Transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64,64)),
    transforms.RandomRotation(degrees=15),
  ])
  return Image_Transformer
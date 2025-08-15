from PIL import Image
import torch
from image_predicter import transformer


def predict_Images(img,models,indexes):
  image = Image.open(img).convert("RGB")
  image  = transformer(image)
  image = image.unsqueeze(0)

  models.eval()
  with torch.no_grad():
    output = models(image)
    _,preds = torch.max(output,dim=1)
  print(f"This is the result without the mdoels {output}")
  return indexes[preds[0].item()]
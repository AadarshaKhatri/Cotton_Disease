from flask import jsonify,Flask,redirect,request,render_template
from model.cotton_cnn_model import IMG_CLASSIFIER_MODEL
import torch
from utils.index import indexes
from utils.transformer import transformer
from PIL import Image



model = IMG_CLASSIFIER_MODEL(input_size=3,hidden_size=10,output_size=len(indexes))

model.load_state_dict(torch.load("modelCotton.pth",map_location=torch.device("cpu")))



# Starting Flask over here 


app = Flask(__name__)

@app.route("/",methods=["GET"])
def index():
  return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def upload():
  print("Predict Hit")
  if request.method == "POST" : 
    user_file = request.files["file"]

    image = Image.open(user_file).convert("RGB")
    transform = transformer()
    image = transform(image)
    image = image.unsqueeze(0)

    model.eval()
    with torch.no_grad():
      output = model(image)
      _,pred = torch.max(output,1)

  return render_template("index.html", prediction=indexes[pred.item()])
    
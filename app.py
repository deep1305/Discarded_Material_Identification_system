import sys, os, shutil, subprocess
from wasteDetection.pipeline.training_pipeline import TrainPipeline
from wasteDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from wasteDetection.constant.application import APP_HOST, APP_PORT


app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"



@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfull!!" 


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        # clean previous runs so output always lands in exp/
        shutil.rmtree("yolov5/runs", ignore_errors=True)

        result = subprocess.run(
            [sys.executable, "detect.py", "--weights", "best.pt",
             "--img", "416", "--conf", "0.5",
             "--source", "../data/inputImage.jpg",
             "--exist-ok"],
            cwd=os.path.join(os.getcwd(), "yolov5"),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("YOLOv5 error:", result.stderr)
            return Response(f"Detection failed: {result.stderr}", status=500)

        # find the saved output image dynamically
        import glob
        matches = glob.glob("yolov5/runs/detect/**/inputImage.jpg", recursive=True)
        if not matches:
            return Response(f"Detection ran but output image not found. stdout: {result.stdout}", status=500)

        opencodedbase64 = encodeImageIntoBase64(matches[0])
        response = {"image": opencodedbase64.decode('utf-8')}
        shutil.rmtree("yolov5/runs", ignore_errors=True)
        return jsonify(response)

    except ValueError as val:
        print(val)
        return Response("Value not found inside json data", status=400)
    except KeyError:
        return Response("Key value error: incorrect key passed", status=400)
    except Exception as e:
        print(e)
        return Response(f"Error: {str(e)}", status=500)



@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        subprocess.run(
            [sys.executable, "detect.py", "--weights", "best.pt",
             "--img", "416", "--conf", "0.5", "--source", "0"],
            cwd=os.path.join(os.getcwd(), "yolov5")
        )
        shutil.rmtree("yolov5/runs", ignore_errors=True)
        return "Camera starting!!" 

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    



if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)


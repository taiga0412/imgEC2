from flask import Flask, jsonify, request
import ast
import json
from PIL import Image
import glob
import cv2
import base64
import numpy as np
import io

from ImageAI import imageAI

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

@app.route("/test", methods=['POST'])
def testPost():

    message = request.get_json()
    res = ast.literal_eval(message)
    res = res["text"]
    print(res)

    

    img_binary = base64.b64decode(res)
    jpg=np.frombuffer(img_binary,dtype=np.uint8)

    img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    #画像を保存する場合
    cv2.imwrite("decode.jpg",img)
    

    
    res = imageAI("decode.jpg")
    return res 


if __name__ == "__main__":
    app.run(host='18.233.166.12',port=5000,debug = True)
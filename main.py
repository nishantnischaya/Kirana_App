from flask import *
from scripts.getDateTime import DateTime
from scripts.data import BillData
from scripts.test import argmax
from scripts.base64conversion import convertToImage
import cv2

app = Flask(__name__)

@app.route('/')
def index():
   TimeStamp = DateTime()
   Date, Time = TimeStamp.split(' ')
   Data = BillData()
   s = 0.0
   for d in Data:
      s += d.get('price') * d.get('quantity')
   return render_template('index.html', date = Date, time = Time, datas = Data, sum = s)

@app.route('/image', methods=['POST'])
def getImage():
   image_obj = request.json
   # print(type(image_obj))
   imagestring = image_obj['data'] 
   s = convertToImage(imagestring)
   image = cv2.imread('temp_image.jpg')
   predicted_value = argmax(image)
   return 'success'

if __name__ == '__main__':
   app.run(host='0.0.0.0',debug=True,port='5000')
  
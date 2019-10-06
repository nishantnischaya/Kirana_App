from flask import *
from scripts.getDateTime import DateTime
from scripts.data import BillData
from scripts.test import argmax

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
   image = image_obj['data'] 
   # print(image)
   return image_obj

if __name__ == '__main__':
   app.run(host='0.0.0.0',debug=True,port='5000')

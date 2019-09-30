from flask import *
from getDateTime import DateTime
from data import BillData

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

if __name__ == '__main__':
   app.run(host='0.0.0.0',debug=True,port='5000')

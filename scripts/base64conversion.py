import base64

def convertToImage(imgstring):
    imgdata = base64.b64decode(imgstring)
    filename = 'temp_image.jpg' 
    with open(filename, 'wb+') as f:
        f.write(imgdata)
    return filename
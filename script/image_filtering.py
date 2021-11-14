import cv2

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions as resdec
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions as incdec

incmodel = InceptionV3()
resmodel = ResNet50(weights='imagenet')

def imageFiltering(img): ## False= 제거해야할 이미지
  incimg=[]
  resimg=[]
  incimg.append(cv2.resize(img, (299,299), interpolation = cv2.INTER_AREA))
  resimg.append(cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA))

  processed = np.array(incimg).astype('float32')
  processed = preprocess_input(processed)
  incyhat = incmodel.predict(processed)

  resyhat = resmodel.predict(np.array(resimg))

  target='web_site'

  incdeco=incdec(incyhat, top=1)
  resdeco=resdec(resyhat, top=1)
  if (incdeco[0][0][1] in target) or (resdeco[0][0][1] in target):
    return False
  else:
    return True
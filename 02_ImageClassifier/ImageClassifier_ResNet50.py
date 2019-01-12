# 画像識別（ResNet50）

# coding:utf-8
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# モデル読み込み（ResNet50）
model = ResNet50(weights='imagenet')

# 画像読み込み
img_path = 'test.jpg'
img = image.load_img(img_path, target_size=(224, 224))

# 前処理
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 予測
preds = model.predict(x)

# 表示
for name, description, score in decode_predictions(preds, top=3)[0]:
    print(description + ": " + str(int(score * 100)) + "%")

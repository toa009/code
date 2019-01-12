# 画像識別（inceptionV3）

from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
import numpy as np

# モデル読み込み（InceptionV3）
model = InceptionV3(weights='imagenet')

# 画像読み込み
img_path = 'test.jpg'
img = image.load_img(img_path, target_size=(299, 299))

# 前処理
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 予測
preds = model.predict(x)

# 表示
for p in decode_predictions(preds, top=5)[0]:
    print("Score {}, Label {}".format(p[2], p[1]))

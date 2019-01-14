# 手書き文字認識（モデル構築用）
from keras.datasets import mnist
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras import optimizers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import glob
import os,sys
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Dropout

# 1. データ取得

# ラベル用辞書作成
dic_katakana = {"a":0,"i":1,"u":2,"e":3,"o":4}
pixel = 28

#元画像を表示させる方法¶  28ピクセル*28ピクセルのサイズで、0から255までのグレースケール画像。
id_ = 0
for katakana in dic_katakana.keys():
    img_ = Image.open("../1_data/train/%s/%s_%s.png"%(katakana,katakana, id_))

#元画像をnumpy形式に変換する方法&前処理の例

# trainデータのファイルパスを取得
li_fpath = glob.glob("../1_data/train/*/*.png")
li_fpath[:10]

# numpy形式に変換
num_image = len(li_fpath)
channel = 1 # グレースケール
train_data = np.empty((num_image, channel, pixel, pixel))
train_label = []

for i, fpath in enumerate(li_fpath):

    label_str = os.path.split(fpath)[1].split("_")[0]
    label_int = dic_katakana[label_str]
    train_label.append(label_int)


    img_ = Image.open(fpath)
    img_ = np.array(img_).astype(np.float32)
    train_data[i, 0, :] = img_

# print("train_data.shape=", train_data.shape)

# 正規化
train_data = train_data / train_data.max()
train_data = train_data.astype('float32')
#print(train_data)

# one hotベクトル化
lb = LabelBinarizer()
train_label =lb.fit_transform(train_label).astype('int32')

#前処理したデータをファイル保存
np.save("../1_data/train_data.npy", train_data)
np.save("../1_data/train_label.npy", train_label)

# 2. モデル構築

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=784))
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(5, activation='softmax'))

#モデルのコンパイル
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 次元変換してfitできるようにする
train_data = train_data.reshape((1000,784))
#train_data.shape

# 3. 学習
history = model.fit(train_data, train_label, verbose=0, epochs=50)

# 4. 評価
score = model.evaluate(train_data, train_label, verbose=1)
print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))

#　プロット
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

# モデル保存　
model.save("model.h5")  

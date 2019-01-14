# 手書き文字認識（予測用）

import keras
from PIL import Image
import numpy as np
import glob
import os,sys
from sklearn.preprocessing import LabelBinarizer

def makedataset(dirpath):

    # testデータのファイルパスを取得
    li_fpath = glob.glob(os.path.join(dirpath, "*","*.png"))

    # 1. データ取得

    # ラベル用辞書作成
    dic_katakana = {"a":0,"i":1,"u":2,"e":3,"o":4}
    pixel = 28

    # 元画像をnumpy形式に変換する方法&前処理の例

    # numpy配列用意
    num_image = len(li_fpath)
    channel = 1 # グレースケール
    train_data = np.empty((num_image, channel, pixel, pixel))
    train_label = []

    for i, fpath in enumerate(li_fpath):

        # ラベル作成
        label_str = os.path.split(fpath)[1].split("_")[0]
        label_int = dic_katakana[label_str]
        train_label.append(label_int)

        # データ作成
        img_ = Image.open(fpath)
        img_ = np.array(img_).astype(np.float32)
        train_data[i, 0, :] = img_

    # 正規化
    train_data = train_data / train_data.max()
    train_data = train_data.astype('float32')

    # one hotベクトル化
    lb = LabelBinarizer()
    train_label =lb.fit_transform(train_label).astype('int32')

    # 次元変換してfitできるようにする
    train_data = train_data.reshape((6500,784))

    test_data = train_data
    test_label = train_label

    return test_data, test_label


def predict(test_data, test_label):

    ## 学習結果を読む(TensoFlowを用いた場合の例)
    model = keras.models.load_model("model.h5")

    ## 予測(TensoFlowを用いた場合の例)
    score = model.evaluate(test_data, test_label, verbose=0)

    return score[0], score[1]


def main(dirpath):

    # テスト用データをつくる
    test_data, test_label = makedataset(dirpath)

    # 予測し、損失と精度を返す
    loss, accuracy = predict(test_data, test_label)

    return loss, accuracy

if __name__=="__main__":

    dirpath = "../1_data/test"
    loss, accuracy = main(dirpath)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

# request, redirect, url_for を追加
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pandas.io.sql as psql
import sqlite3
from contextlib import closing

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db0.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.Integer) # , unique=True
    age = db.Column(db.Integer) #
    sex = db.Column(db.Integer) #
    label = db.Column(db.Integer) #
    #id = db.Column(db.Integer)
    #username = db.Column(db.Integer)

    def __init__(self, username, age, sex, label):
        self.username = username
        self.age = age #
        self.sex = sex #
        self.label = label #

a = ""
n = 100

@app.route("/")
def hello():
    user_list = User.query.all()
    global a

    return render_template('hello.html', user_list=user_list, a=a)

# htmlからフォームデータを受け取る。add_user メソッドを追加し、ユーザ登録に関する処理を行う
@app.route("/add_user", methods=['POST'])
def add_user():

    # フォームから送られてきた age等 を取得
    #username = request.form.get('username')
    age = request.form.get('age')
    sex = request.form.get('sex')
    label = request.form.get('label')

    # usernameはユニークの必要がある（なぜか）ので、python内で適当に与える（フォームデータは無視）
    global n
    username = n
    n += 1

    if username:
        # 前回、手動で対応した処理と同じ
        user1 = User(username, age, sex, label) # Userインスタンス（user）生成
        db.session.add(user1)
        db.session.commit() # 反映

    # ユーザ登録後は、元ページへリダイレクト
    return redirect(url_for('hello'))

@app.route("/clear", methods=['POST'])
def clear():

    # usernameはユニークの必要がある（なぜか）ので、python内で適当に与える（フォームデータは無視）
    global n
    username = n
    n += 1

    # DBのテーブル内容を全て削除
    #db.session.query(User).filter( User.id == 100 ).delete()
    db.session.query(User).delete()
    #user1 = User(username=101) # Userインスタンス（user）生成
    #db.session.delete(user1)
    db.session.commit() # 反映

    # 元ページへリダイレクト
    return redirect(url_for('hello'))

@app.route("/predict", methods=['POST'])
def predict():
    global a
    a = "predict init"

    # フォームからageを取得
    age = request.form.get('age')
    sex = request.form.get('sex')

    #---------------------------
    dbname = 'db0.db'

    with closing(sqlite3.connect(dbname)) as conn:
        c = conn.cursor()
        sql = 'select * from user'
        df = psql.read_sql(sql, conn)

        # モデリング
        ax = df.loc[:,['age','sex']]
        ay = df.loc[:,['label']]
        #ex = [[0,0,],[0,1],[1,0],[1,1],[3,3],[4,4],[5,5]]
        ex = [[float(age),float(sex)]]

        #from sklearn.ensemble import RandomForestClassifier
        #m = RandomForestClassifier()
        from sklearn.linear_model import LinearRegression
        m = LinearRegression()

        m.fit(ax, ay)

        # 予測
        py = m.predict(ex)
        #py = m.predict([[1,1]])

    #----------------------
    #a = "test"
    #a = str(py)[1] # 数値を文字列化　＋　文字列が[1]の形なので2文字目をスライスして数字だけ取得
    a = round(py[0][0],2)

    return redirect(url_for('hello'))

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')

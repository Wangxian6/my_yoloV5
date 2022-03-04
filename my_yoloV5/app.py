import flask
from flask import Flask,jsonify,redirect,url_for,request
import detect

app=Flask(__name__)

@app.route("/bbox")
def bbox(b):
    print(b[3],123)
    return jsonify(b)
@app.route("/")
def main(succ=False):
    opt=detect.parse_opt()
    detect.main(opt)
    if succ:
        return redirect(url_for("bbox"))

if __name__ == '__main__':
    app.run()
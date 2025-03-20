from flask import Flask, render_template

server = Flask(__name__)

@server.route('/')
def index():
    return render_template("index.html")

@server.route('/predict-page')
def predict_page():
    return render_template("predict.html")

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=8000)
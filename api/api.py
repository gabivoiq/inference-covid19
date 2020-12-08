from flask import Flask, render_template, request
from main.test import process_data

app = Flask(__name__)


@app.route('/upload')
def upload_file_html():
    return render_template("upload.html")


@app.route('/uploader', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        return process_data(f)


app.run()

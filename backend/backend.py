from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_data():
    try:
        model_video = request.files['modelVideo']
        student_video = request.files['studentVideo']
        tips_text = request.form['tipsText']

        # ここで受け取ったデータを処理する

        response_data = {'status': 'success', 'message': 'Data received successfully'}
        return jsonify(response_data)

    except Exception as e:
        response_data = {'status': 'error', 'message': str(e)}
        return jsonify(response_data), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

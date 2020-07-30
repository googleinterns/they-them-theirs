from flask import Flask, jsonify, request, render_template
from flask_bootstrap import Bootstrap

import sys

sys.path.append('../neutral_generation/')
from smart_convert import convert

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/gender_neutral')
def gender_neutral():
    try:
        sentence = request.args.get('sentence', 0, type=str)
        neutral_sentence = convert(sentence)
        results = {'original_sentence': sentence,
                   'neutral_sentence': neutral_sentence}
        return jsonify(results)

    except Exception as e:
        return e


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# A list to store user-submitted names
user_names = []

@app.route('/')
def home():
    return 'Welcome to Flask Example!'

@app.route('/inference', methods=['GET', 'POST'])
def add_name():
    if request.method == 'POST':
            image_path = request.form['ImagePath']
            model_weights_path = request.form['ModelWeights']
            print(image_path)
            print(model_weights_path)
            return render_template('inference_results.html')
    return render_template('inference.html')

if __name__ == '__main__':
    app.run(debug=True)
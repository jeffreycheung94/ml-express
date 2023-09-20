from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# A list to store user-submitted names
user_names = []

@app.route('/')
def home():
    return 'Welcome to Flask Example!'

@app.route('/add_name', methods=['GET', 'POST'])
def add_name():
    if request.method == 'POST':
        name = request.form['name']
        if name:
            user_names.append(name)
            return redirect(url_for('list_names'))
    return render_template('add_name.html')

@app.route('/list_names')
def list_names():
    return render_template('list_names.html', names=user_names)

if __name__ == '__main__':
    app.run(debug=True)
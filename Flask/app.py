from flask import Flask, request, jsonify, render_template

app = Flask(__name__, static_folder='static')

# Replace with your BERT model loading and prediction code
def predict_revenue(prompt, budget, country, language):
    # print(prompt, budget, country, language)
    # This is dummy data for illustration purposes
    return [
        {"country": "USA", "revenue": budget*5},
        {"country": "UK", "revenue": budget*2},
        {"country": "India", "revenue": budget*1.5},
    ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    prompt = data['prompt']
    budget = int(data['budget'])
    country = data['country']
    language = data['language']

    revenueData = predict_revenue(prompt, budget, country, language)
    
    return jsonify({'revenueData': revenueData})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Set use_reloader to False to avoid duplicate output on some systems
    print(f"Running on {app.config['SERVER_NAME']}")
from flask import Flask, request, jsonify, render_template
# from utils import predict_revenues
from genre_prediction import GenrePredictor

app = Flask(__name__, static_folder='static')
genre_predictor = GenrePredictor()

# Replace with your BERT model loading and prediction code
def predict_revenue(prompt, budget, language):
    # print(prompt, budget, country, language)
    # This is dummy data for illustration purposes
    return [
        {"country": "USA", "revenue": budget*5},
        {"country": "UK", "revenue": budget*2},
        {"country": "India", "revenue": budget*1.5},
    ]

# predict_revenues(prompt, original_language, genres, budget)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster')
def index_cluster():
    return render_template('cluster.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    prompt = data['prompt']
    budget = int(data['budget'])
    # country = data['country']
    language = data['language']

    revenueData =  predict_revenue(prompt, budget, language) #predict_revenues(prompt, language, [], budget)
    
    return jsonify({'revenueData': revenueData})

# Add endpoint for /api/genres to return the list of genres predicted by the model
@app.route('/api/genres', methods=['POST'])
def predict_genres():
    data = request.get_json()
    prompt = data['plot']
    predicted_genres = genre_predictor.predict_genres(prompt)
    return jsonify({'genres': predicted_genres})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Set use_reloader to False to avoid duplicate output on some systems
    print(f"Running on {app.config['SERVER_NAME']}")
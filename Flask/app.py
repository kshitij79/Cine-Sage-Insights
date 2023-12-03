from flask import Flask, request, jsonify, render_template
from genre_prediction import GenrePredictor
from countrywise_revenue_predictor import WorldWideRevenuePredictor
from overall_revenue_predictor import OverallRevenuePredictor
# from utils import inverted_language_mapping

app = Flask(__name__, static_folder='static')
genre_predictor = GenrePredictor()
overall_revenue_predictor = OverallRevenuePredictor()
worldwide_revenue_predictor = WorldWideRevenuePredictor()

# # Replace with your BERT model loading and prediction code
# def predict_revenue(prompt, budget, language):
#     # print(prompt, budget, country, language)
#     # This is dummy data for illustration purposes
#     return [
#         {"country": "USA", "revenue": budget*5},
#         {"country": "UK", "revenue": budget*2},
#         {"country": "India", "revenue": budget*1.5},
#     ]

# def random_predict_revenue(prompt, budget, language):

#     if language not in inverted_language_mapping:
#         language = "en"
#     else:    
#         original_language = inverted_language_mapping[language]
        
#     print(prompt, budget, original_language)
#     language_to_countries = {
#         "en": ["USA", "UK", "India", "Australia", "Canada"],
#         "es": ["Argentina", "Mexico", "Spain"],
#         "fr": ["France", "Canada"],
#         "de": ["Germany", "Austria", "Switzerland"],
#         "ko": ["South Korea"],
#         "zh": ["Taiwan"],
#         "hi": ["India"],
#         "pt": ["Brazil", "Portugal"]
#         # Add more languages and countries as needed
#     }

#     eligible_countries = language_to_countries.get(original_language, ["USA", "UK", "India"])

#     # Randomly selecting countries and calculating revenues
#     random_countries = random.sample(eligible_countries, k=len(eligible_countries))
#     results = []
#     for country in random_countries:
#         if country == "USA":
#             revenue = random.uniform(4, 6) * budget  # Random multiplier between 4 and 6
#         elif country == "UK":
#             revenue = random.uniform(1.5, 2.5) * budget  # Random multiplier between 1.5 and 2.5
#         elif country == "India":
#             revenue = random.uniform(1, 2) * budget  # Random multiplier between 1 and 2
#         else:
#             revenue = random.uniform(0.5, 1.5) * budget  # Default case for other countries

#         results.append({"country": country, "revenue": revenue})
#     print(results)
#     return results


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
    
    movie_data = {
        'overview': data['prompt'],
        'budget': int(data['budget']),
        'original_language': data['language'],
        'genre_list': data['genres']
    }
    revenue_data = []
    overall_revenue = int(overall_revenue_predictor.predict_revenue(movie_data))
    for country, revenue in worldwide_revenue_predictor.predict_revenues(movie_data, overall_revenue).items():
        revenue_data.append({'country': country, 'revenue': revenue})
    revenue_data.append({'country': 'Overall', 'revenue': overall_revenue})
    return jsonify({'revenueData': revenue_data})

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
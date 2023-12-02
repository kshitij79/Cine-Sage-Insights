import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class GenrePredictor:
    def __init__(self, sigmoid_threshold=0.5):
        self.sigmoid_threshold = sigmoid_threshold
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("static/genre_prediction_model")
        self.genre2id = {'genre_Action': 0,
            'genre_Adventure': 1,
            'genre_Animation': 2,
            'genre_Comedy': 3,
            'genre_Crime': 4,
            'genre_Documentary': 5,
            'genre_Drama': 6,
            'genre_Family': 7,
            'genre_Fantasy': 8,
            'genre_Foreign': 9,
            'genre_History': 10,
            'genre_Horror': 11,
            'genre_Music': 12,
            'genre_Mystery': 13,
            'genre_Romance': 14,
            'genre_Science Fiction': 15,
            'genre_TV Movie': 16,
            'genre_Thriller': 17,
            'genre_War': 18,
            'genre_Western': 19
        }
        self.id2genre = {v: k.split('_')[1] for k, v in self.genre2id.items()}
        self.genres = list(self.id2genre.values())

    def predict_genres(self, plot_overview):
        # Tokenize input
        inputs = self.tokenizer(plot_overview, truncation=True, padding=True, return_tensors="pt")
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        # Make prediction
        outputs = self.model(**inputs)
        probs = torch.nn.Sigmoid()(outputs.logits)
        preds = (probs >= self.sigmoid_threshold).long().cpu().numpy()
        predicted_genres = [self.id2genre[i] for i, pred in enumerate(preds[0]) if pred == 1]
        return predicted_genres


if __name__ == '__main__':
    test_movies = [
        {
            'title': 'Oppenheimer',
            'overview': "The story of American scientist, J. Robert Oppenheimer, and his role in the development of the atomic bomb.",
            'genre_list': ['Drama', 'History', 'Thriller']
        },
        {
            'title': 'Barbie',
            'overview': "Barbie suffers a crisis that leads her to question her world and her existence.",
            'genre_list': ['Adventure', 'Comedy', 'Fantasy']
        },
        {
            'title': 'Everything Everywhere All at Once',
            'overview': "A middle-aged Chinese immigrant is swept up into an insane adventure in which she alone can save existence by exploring other universes and connecting with the lives she could have led.",
            'genre_list': ['Action', 'Adventure', 'Comedy']
        }
    ]
    predictor = GenrePredictor()
    for movie in test_movies:
        print(f"Predicting genres for {movie['title']}")
        predicted_genres = predictor.predict_genres(movie['overview'])
        print(f"Predicted genres: {predicted_genres}")
        print(f"Actual genres: {movie['genre_list']}")
import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer
import pandas as pd
import json

MODEL_DIR = 'static/saved_models/'

TOP_CAST = pd.read_csv('static/top_cast_country_wise.csv')
TOP_CREW = pd.read_csv('static/top_crew_country_wise.csv')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COUNTRIES = ['Argentina', 'Australia', 'Austria', 'Belgium', 'Domestic', 
             'South Korea', 'Spain', 'Taiwan', 'United Kingdom', 'France', 
            #  'Germany', 
            #  'Italy',
            #  'Mexico',
            #  'Netherlands'
             ]
COUNTRY_WISE_CONFIG_PATH = 'static/country_wise_config.json'

class CountrywiseRevenuePredictorModel(nn.Module):
    def __init__(self,
            bert_embedding_size,
            cast_embedding_size,
            crew_embedding_size,
            hidden_size,
            num_cast,
            num_crew,
            num_genres,
            num_original_languages
        ):
        super(CountrywiseRevenuePredictorModel, self).__init__()
        self.num_cast = num_cast
        self.num_crew = num_crew
        self.num_genres = num_genres
        self.num_original_languages = num_original_languages
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Linear layer for textual embeddings
        self.linear_overview = nn.Linear(self.bert.config.hidden_size, bert_embedding_size)

        # # Linear layer for original language embeddings
        # self.linear_original_language = nn.Linear(NUM_ORIGINAL_LANGUAGES, original_language_embedding_size)

        # Linear layer for embedding cast
        self.linear_cast = nn.Linear(num_cast, cast_embedding_size)

        # Linear layer for embedding crew
        self.linear_crew = nn.Linear(num_crew, crew_embedding_size)

        # Budget and budget_unknown, and genres
        self.other_features_size = 2 + num_genres + num_original_languages

        self.output_layer = nn.Sequential(
            nn.Linear(bert_embedding_size + cast_embedding_size + crew_embedding_size + self.other_features_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input):
        bert_out = self.bert(input_ids=input[:, :256].long(), attention_mask=input[:, 256:512].long())
        overview_embedding = self.linear_overview(bert_out['pooler_output'])
        overview_embedding = nn.LeakyReLU()(overview_embedding)

        original_language = input[:, 512:512+self.num_original_languages]
        cast_embedding = self.linear_cast(input[:, 512+self.num_original_languages:512+self.num_original_languages+self.num_cast])
        cast_embedding = nn.LeakyReLU()(cast_embedding)
        crew_embedding = self.linear_crew(input[:, 512+self.num_original_languages+self.num_cast:512+self.num_original_languages+self.num_cast+self.num_crew])
        cast_embedding = nn.LeakyReLU()(cast_embedding)
        other_features = input[:, 512+self.num_original_languages+self.num_cast+self.num_crew:]

        return self.output_layer(torch.cat((
            overview_embedding,
            original_language,
            cast_embedding,
            crew_embedding,
            other_features
        ), dim=1))

class CountrywiseRevenuePredictor():
    def __init__(self, model_config, top_cast_df, top_crew_df):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.revenue_scale = model_config['revenue_scale']
        self.genre_cols = model_config['genre_cols']
        self.num_genres = len(self.genre_cols)
        self.cast_cols = model_config['cast_cols']
        self.num_cast = len(self.cast_cols)
        self.crew_cols = model_config['crew_cols']
        self.num_crew = len(self.crew_cols)
        self.original_language_cols = model_config['original_language_cols']
        self.num_original_languages = len(self.original_language_cols)
        self.model = CountrywiseRevenuePredictorModel(
            bert_embedding_size=model_config['model_params']['bert_hidden_size'],
            cast_embedding_size=model_config['model_params']['cast_size'],
            crew_embedding_size=model_config['model_params']['crew_size'],
            hidden_size=model_config['model_params']['hidden_size'],
            num_cast=self.num_cast,
            num_crew=self.num_crew,
            num_genres=self.num_genres,
            num_original_languages=self.num_original_languages
        )
        self.model.load_state_dict(torch.load(
            MODEL_DIR + model_config['model_path'],
            map_location=DEVICE
        ))
        self.model.eval()
        self.top_cast = self.prepare_name_to_col_dict(top_cast_df, self.cast_cols)
        self.top_crew = self.prepare_name_to_col_dict(top_crew_df, self.crew_cols)
    
    def prepare_name_to_col_dict(self, df, cols_order):
        result = {}
        for i, row in df.iterrows():
            if row['id'] in cols_order:
                result[row['name']] = cols_order.index(row['id'])
        return result
    
    def get_cast_tensor(self, cast_names):
        cast_tensor = torch.zeros(self.num_cast, device=DEVICE)
        for name in cast_names:
            if name in self.top_cast:
                cast_tensor[self.top_cast[name]] = 1
        return cast_tensor

    def get_crew_tensor(self, crew_names):
        crew_tensor = torch.zeros(self.num_crew, device=DEVICE)
        for name in crew_names:
            if name in self.crew_cols:
                crew_tensor[self.crew_cols.index(name)] = 1
        return crew_tensor

    def get_genre_tensor(self, genres):
        genre_tensor = torch.zeros(self.num_genres, device=DEVICE)
        for genre in genres:
            if genre in self.genre_cols:
                genre_tensor[self.genre_cols.index(genre)] = 1
        return genre_tensor
    
    def predict_revenue(self, movie_data):
        '''
        movie_data: list of dictionaries, each dictionary contains the following keys:
            - overview: string, overview of the movie, required
            - genre_list: list of strings, genres of the movie, required
            - original_language: string, original language of the movie (optional)
            - cast: list of strings, names of the cast (optional)
            - crew: list of strings, names of the crew (optional)
            - budget: int, budget of the movie (optional)
        '''
        encoded = self.tokenizer.encode_plus(
            movie_data['overview'], add_special_tokens=True,
            max_length=256, padding='max_length', truncation=True, return_tensors='pt'
        ).to(DEVICE)
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        original_language = torch.zeros(self.num_original_languages, device=DEVICE)
        original_language[self.original_language_cols.index(
            movie_data['original_language'] if 'original_language' in movie_data else 'en'
        )] = 1
        genres = self.get_genre_tensor(movie_data['genre_list'] if 'genre_list' in movie_data else [])
        cast = self.get_cast_tensor(movie_data['cast'] if 'cast' in movie_data else [])
        crew = self.get_crew_tensor(movie_data['crew'] if 'crew' in movie_data else [])
        if 'budget' in movie_data and movie_data['budget'] != 0:
            budget = torch.tensor([movie_data['budget'] / self.revenue_scale], device=DEVICE)
            budget_unknown = torch.tensor([0], device=DEVICE)
        else:
            budget = torch.tensor([0], device=DEVICE)
            budget_unknown = torch.tensor([1], device=DEVICE)
        input = torch.cat((
            input_ids, attention_mask, original_language, genres, cast, crew, budget, budget_unknown
        )).unsqueeze(0)
        prediction = self.model(input).item()
        return prediction * self.revenue_scale * 1e6

class WorldWideRevenuePredictor():
    def __init__(self):
        config = json.load(open(COUNTRY_WISE_CONFIG_PATH, 'r'))
        self.countrywise_predictors = {}
        for country in COUNTRIES:
            print("Loading model for ", country) 
            self.countrywise_predictors[country] = CountrywiseRevenuePredictor(config[country], TOP_CAST, TOP_CREW)

    def predict_revenues(self, movie_data, total_revenue):
        '''
        movie_data: list of dictionaries, each dictionary contains the following keys:
            - overview: string, overview of the movie, required
            - genre_list: list of strings, genres of the movie, required
            - original_language: string, original language of the movie (optional)
            - cast: list of strings, names of the cast (optional)
            - crew: list of strings, names of the crew (optional)
            - budget: int, budget of the movie (optional)
        '''
        revenues = {}
        def cap(rev, limit):
            # If rev is greater than limit, convert rev to [0, 1] and scale to limit
            return rev if rev < limit else (limit * rev / (10 ** len(str(int(rev)))))
        
        sum_revenues = 0
        for country in COUNTRIES:
            predicted_revenue = abs(self.countrywise_predictors[country].predict_revenue(movie_data))
            result_country = country
            if country == 'Domestic':
                result_country = 'United States of America'
                predicted_revenue = cap(predicted_revenue, 1e9)
            elif country == 'South Korea':
                result_country = 'Korea, South'
                predicted_revenue = cap(predicted_revenue, 1e8)
            elif predicted_revenue > 1e9:
                predicted_revenue = cap(predicted_revenue, 1e8)
            revenues[result_country] = int(predicted_revenue)
            sum_revenues += int(predicted_revenue)
        if sum_revenues >= total_revenue:
            scale = 0.9 * total_revenue / sum_revenues
            for country in revenues:
                revenues[country] = int(revenues[country] * scale)
        return revenues

if __name__ == '__main__':
    predictor = WorldWideRevenuePredictor()
    test_movies = [
        {
            'title': 'Oppenheimer',
            'overview': "The story of American scientist, J. Robert Oppenheimer, and his role in the development of the atomic bomb.",
            'genre_list': ['Drama', 'History', 'Thriller'],
            'original_language': 'en',
            'budget': 1e8,
            'cast': ['Cillian Murphy', 'Emily Blunt', 'Matt Damon', 'Robert Downey Jr.'],
            'crew': ['Christopher Nolan']
        },
        {
            'title': 'Barbie',
            'overview': "Barbie suffers a crisis that leads her to question her world and her existence.",
            'genre_list': ['Adventure', 'Comedy', 'Fantasy'],
            'original_language': 'en',
            'budget': 1.45e8,
            'cast': ['Margot Robbie', 'Emma Mackey', 'Ryan Gosling', 'Issa Rae'],
            'crew': ['Greta Gerwig']
        },
        {
            'title': 'Everything Everywhere All at Once',
            'overview': "A middle-aged Chinese immigrant is swept up into an insane adventure in which she alone can save existence by exploring other universes and connecting with the lives she could have led.",
            'genre_list': ['Action', 'Adventure', 'Comedy'],
            'original_language': 'en',
            'budget': 25e6,
            'cast': ['Michelle Yeoh', 'Stephanie Hsu', 'Jamie Lee Curtis'],
            'crew': ['Daniel Kwan', 'Daniel Scheinert']
        }
    ]
    for movie in test_movies:
        print(movie['title'])
        print(predictor.predict_revenues(movie))
        print()
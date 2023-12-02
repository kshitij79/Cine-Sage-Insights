
# import tokenize
# import torch
# from transformers import BertTokenizer 
# import torch.nn as nn
# from model import RevenuePredictor  # Import your model class

# NUM_GENRES = 20
# NUM_CAST = 0
# NUM_CREW = 0
# NUM_ORIGINAL_LANGUAGES = 90
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def predict_revenues(prompt, original_language, genres, budget, device=DEVICE):
#     inputs = []
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     model = RevenuePredictor(bert_embedding_size=256, original_language_embedding_size=16, hidden_size=128, cast_embedding_size=64, crew_embedding_size=16)
#     model.load_state_dict(torch.load('best_model_overall_revenue.pth', map_location=DEVICE))
#     encoded = tokenizer.encode_plus(prompt, add_special_tokens=True, max_length=256, padding='max_length', truncation=True, return_tensors='pt').to(device)
#     input_ids = encoded["input_ids"].squeeze()
#     attention_mask = encoded["attention_mask"].squeeze()

#     crew_tensor = torch.zeros(NUM_CREW, device=DEVICE)
#     cast_tensor = torch.zeros(NUM_CAST, device=DEVICE)
#     genres = torch.zeros(20, device=DEVICE)

#     # genres = get_genre_tensor(movie['genre_list'])

#     language_one_hot = [0] * NUM_ORIGINAL_LANGUAGES
#     # language_one_hot[train_dataset.original_language_cols.index('original_language_' + movie['original_language'])] = 1
#     original_language = torch.tensor(language_one_hot, dtype=torch.float).to(device)

#     # cast = get_cast_tensor(movie['cast'])
#     # crew = get_crew_tensor(movie['crew'])

#     if budget is not None:
#         budget_tensor = torch.tensor([budget / 1e8], dtype=torch.float).to(device)
#         budget_unknown = torch.tensor([0], dtype=torch.float).to(device)
#     else:
#         budget_tensor = torch.tensor([0], dtype=torch.float).to(device)
#         budget_unknown = torch.tensor([1], dtype=torch.float).to(device)

#     inputs.append(torch.cat((
#         input_ids,
#         attention_mask,
#         original_language,
#         cast_tensor,
#         crew_tensor,
#         genres,
#         budget_tensor,
#         budget_unknown
#     )))

#     inputs = torch.stack(inputs)

#     # Predict
#     with torch.no_grad():
#         prediction = model(inputs)

#     return (1e8 * prediction).tolist()


# language_list = [
#     'ab', 'af', 'am', 'ar', 'ay', 'bg', 'bm', 'bn', 'bo', 'bs', 'ca', 'cn', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo',
#     'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'gl', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'is', 'it', 'iu', 'ja', 'jv', 'ka',
#     'kk', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'nb', 'ne', 'nl', 'no',
#     'pa', 'pl', 'ps', 'pt', 'qu', 'ro', 'ru', 'rw', 'sh', 'si', 'sk', 'sl', 'sm', 'sq', 'sr', 'sv', 'ta', 'te', 'tg', 'th',
#     'tl', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xx', 'zh', 'zu', 'nan'
# ]

# language_mapping = {
#         'ab': 'Abkhazian', 'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'ay': 'Aymara', 'bg': 'Bulgarian',
#         'bm': 'Bambara', 'bn': 'Bengali', 'bo': 'Tibetan', 'bs': 'Bosnian', 'ca': 'Catalan', 'cn': 'Chinese',
#         'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto',
#         'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'fi': 'Finnish', 'fr': 'French', 'fy': 'Frisian',
#         'gl': 'Galician', 'he': 'Hebrew', 'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian', 'hy': 'Armenian', 'id': 'Indonesian',
#         'is': 'Icelandic', 'it': 'Italian', 'iu': 'Inuktitut', 'ja': 'Japanese', 'jv': 'Javanese', 'ka': 'Georgian',
#         'kk': 'Kazakh', 'kn': 'Kannada', 'ko': 'Korean', 'ku': 'Kurdish', 'ky': 'Kyrgyz', 'la': 'Latin', 'lb': 'Luxembourgish',
#         'lo': 'Lao', 'lt': 'Lithuanian', 'lv': 'Latvian', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian',
#         'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'nb': 'Norwegian Bokmål', 'ne': 'Nepali', 'nl': 'Dutch',
#         'no': 'Norwegian', 'pa': 'Punjabi', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'qu': 'Quechua',
#         'ro': 'Romanian', 'ru': 'Russian', 'rw': 'Kinyarwanda', 'sh': 'Serbo-Croatian', 'si': 'Sinhalese', 'sk': 'Slovak',
#         'sl': 'Slovenian', 'sm': 'Samoan', 'sq': 'Albanian', 'sr': 'Serbian', 'sv': 'Swedish', 'ta': 'Tamil', 'te': 'Telugu',
#         'tg': 'Tajik', 'th': 'Thai', 'tl': 'Tagalog', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek',
#         'vi': 'Vietnamese', 'wo': 'Wolof', 'xx': 'Unknown', 'zh': 'Chinese', 'zu': 'Zulu', 'nan': 'Nan'
#     }

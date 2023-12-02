import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer
import pandas as pd

OVERALL_CAST_COLS_ORDER = [3, 4, 13, 19, 31, 32, 34, 35, 48, 49, 50, 52, 53, 62, 63, 64, 65, 85, 99, 103, 109, 110, 112, 113, 114, 116, 118, 131, 133, 134, 139, 140, 141, 147, 190, 192, 193, 204, 205, 206, 207, 209, 212, 227, 228, 230, 258, 287, 290, 326, 335, 349, 350, 352, 368, 378, 380, 382, 388, 400, 418, 452, 454, 477, 480, 500, 501, 504, 514, 515, 516, 517, 518, 519, 520, 521, 522, 524, 526, 529, 532, 534, 537, 539, 540, 569, 585, 588, 589, 591, 649, 650, 655, 658, 677, 707, 723, 738, 776, 819, 821, 824, 827, 854, 879, 880, 882, 883, 884, 886, 887, 920, 923, 925, 934, 935, 937, 938, 955, 973, 976, 980, 1003, 1004, 1018, 1019, 1032, 1037, 1038, 1039, 1062, 1064, 1100, 1117, 1118, 1121, 1125, 1137, 1146, 1158, 1160, 1162, 1164, 1166, 1204, 1205, 1206, 1210, 1211, 1229, 1230, 1231, 1233, 1241, 1243, 1244, 1245, 1246, 1248, 1269, 1271, 1276, 1283, 1284, 1292, 1294, 1327, 1328, 1331, 1333, 1336, 1341, 1370, 1371, 1461, 1462, 1466, 1532, 1533, 1535, 1579, 1639, 1640, 1646, 1653, 1665, 1666, 1733, 1736, 1748, 1771, 1772, 1811, 1812, 1813, 1834, 1844, 1892, 1893, 1896, 1897, 1902, 1903, 1907, 1910, 1920, 1922, 1923, 1925, 1926, 1951, 1954, 1979, 1981, 1989, 2037, 2039, 2047, 2048, 2053, 2090, 2130, 2157, 2165, 2171, 2176, 2178, 2192, 2201, 2203, 2206, 2207, 2219, 2220, 2222, 2224, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2282, 2283, 2295, 2296, 2299, 2314, 2369, 2372, 2387, 2395, 2415, 2440, 2441, 2453, 2461, 2467, 2505, 2524, 2535, 2541, 2561, 2628, 2632, 2641, 2712, 2714, 2778, 2838, 2876, 2879, 2880, 2882, 2887, 2888, 2954, 2955, 2956, 2963, 2969, 2975, 2983, 3019, 3036, 3037, 3041, 3051, 3052, 3061, 3063, 3064, 3085, 3087, 3092, 3124, 3127, 3129, 3131, 3136, 3141, 3151, 3196, 3197, 3201, 3223, 3234, 3265, 3266, 3291, 3293, 3391, 3392, 3416, 3489, 3490, 3491, 3547, 3636, 3713, 3784, 3801, 3872, 3894, 3895, 3896, 3897, 3905, 3910, 3911, 3926, 3967, 3968, 3977, 3982, 4001, 4004, 4029, 4038, 4090, 4135, 4139, 4173, 4175, 4238, 4250, 4252, 4253, 4273, 4390, 4391, 4430, 4431, 4432, 4443, 4451, 4455, 4483, 4491, 4492, 4495, 4496, 4512, 4513, 4515, 4520, 4521, 4566, 4581, 4587, 4589, 4687, 4688, 4690, 4691, 4724, 4726, 4756, 4757, 4764, 4765, 4783, 4784, 4785, 4800, 4808, 4935, 4937, 4941, 4942, 5048, 5049, 5064, 5081, 5139, 5149, 5151, 5168, 5170, 5251, 5274, 5292, 5293, 5294, 5309, 5344, 5365, 5442, 5444, 5469, 5470, 5472, 5502, 5530, 5538, 5563, 5576, 5578, 5587, 5606, 5657, 5658, 5694, 5723, 5724, 5726, 5916, 5950, 5960, 6008, 6012, 6020, 6065, 6104, 6161, 6162, 6164, 6168, 6181, 6193, 6194, 6197, 6199, 6217, 6283, 6352, 6355, 6383, 6384, 6407, 6413, 6450, 6473, 6474, 6486, 6497, 6541, 6573, 6574, 6588, 6613, 6677, 6751, 6752, 6807, 6832, 6837, 6844, 6856, 6885, 6886, 6905, 6908, 6913, 6914, 6941, 6944, 6949, 6952, 6968, 6972, 7004, 7026, 7036, 7056, 7060, 7132, 7166, 7180, 7248, 7399, 7401, 7404, 7447, 7470, 7489, 7497, 7502, 7505, 7517, 7570, 7624, 7633, 7693, 7796, 7868, 7904, 7906, 7907, 7908, 8167, 8170, 8210, 8211, 8212, 8256, 8265, 8289, 8293, 8329, 8335, 8349, 8396, 8435, 8436, 8437, 8447, 8534, 8654, 8655, 8691, 8767, 8783, 8784, 8785, 8789, 8854, 8874, 8891, 8893, 8924, 8930, 8944, 8945, 8949, 8977, 8984, 8986, 9015, 9029, 9030, 9046, 9048, 9137, 9139, 9188, 9191, 9208, 9273, 9278, 9281, 9464, 9560, 9599, 9626, 9629, 9642, 9657, 9777, 9778, 9779, 9780, 9824, 9827, 9880, 9994, 10017, 10127, 10132, 10182, 10205, 10297, 10360, 10401, 10430, 10431, 10437, 10556, 10696, 10697, 10698, 10713, 10767, 10774, 10814, 10822, 10823, 10825, 10859, 10860, 10872, 10959, 10978, 10981, 10983, 10985, 11006, 11064, 11066, 11085, 11086, 11107, 11108, 11148, 11150, 11155, 11159, 11160, 11164, 11181, 11207, 11275, 11355, 11356, 11357, 11367, 11390, 11398, 11477, 11486, 11511, 11512, 11514, 11616, 11661, 11662, 11664, 11701, 11703, 11705, 11805, 11826, 11851, 11857, 11864, 11866, 11870, 11885, 12021, 12041, 12052, 12074, 12077, 12132, 12438, 12519, 12536, 12640, 12647, 12797, 12799, 12834, 12835, 12850, 12852, 12900, 12950, 13014, 13022, 13023, 13240, 13242, 13275, 13333, 13472, 13525, 13548, 13549, 13550, 13688, 13726, 13920, 13922, 13936, 14061, 14324, 14329, 14406, 14415, 14702, 14721, 14792, 14812, 14887, 14888, 15033, 15086, 15091, 15111, 15152, 15234, 15250, 15274, 15277, 15416, 15440, 15498, 15661, 15735, 15758, 15824, 15831, 15852, 15854, 15860, 15900, 16165, 16180, 16327, 16431, 16433, 16475, 16483, 16644, 16828, 16851, 16861, 16866, 16927, 16940, 17039, 17051, 17140, 17141, 17142, 17178, 17199, 17276, 17286, 17287, 17288, 17289, 17328, 17401, 17402, 17419, 17485, 17488, 17604, 17605, 17606, 17696, 17697, 17764, 17772, 17773, 17782, 17832, 17881, 17882, 18023, 18050, 18056, 18071, 18082, 18177, 18262, 18269, 18277, 18284, 18288, 18324, 18325, 18471, 18472, 18686, 18792, 18897, 18916, 18918, 18979, 18992, 18999, 19152, 19159, 19274, 19278, 19292, 19453, 19492, 19839, 19866, 20070, 20089, 20189, 20212, 20243, 20285, 20387, 20519, 20664, 20750, 20753, 20766, 20810, 20904, 20982, 21007, 21028, 21088, 21089, 21104, 21125, 21127, 21163, 21197, 21200, 21278, 21315, 21523, 21657, 21731, 22131, 22132, 22226, 22250, 22306, 22970, 23532, 23626, 23659, 23709, 23880, 23931, 24041, 24045, 24891, 25072, 25246, 25251, 25503, 25540, 26457, 26467, 26472, 26473, 26485, 26510, 26724, 27740, 28010, 28164, 28463, 28633, 28638, 28640, 28641, 28782, 28848, 29369, 29685, 29839, 30613, 31028, 32597, 32747, 33161, 33192, 33517, 33533, 34847, 35070, 35742, 35756, 35776, 35779, 35780, 35793, 35819, 36422, 36594, 36666, 36801, 37233, 37917, 38334, 38559, 38582, 38673, 38940, 39388, 39658, 40462, 40481, 40543, 41087, 41091, 42157, 42802, 42803, 42993, 43120, 43366, 43775, 44079, 44735, 45152, 46691, 47820, 47879, 50463, 51072, 51329, 51576, 51944, 52374, 52404, 52792, 52848, 52995, 52997, 53256, 53650, 53714, 54812, 55152, 55536, 55636, 55638, 56024, 56322, 56731, 56734, 56861, 56890, 57599, 57755, 57829, 58019, 58224, 58225, 58478, 58563, 59315, 59841, 60279, 60650, 60875, 60949, 61659, 61981, 62862, 63585, 64436, 64856, 65717, 65731, 65827, 66717, 67773, 68180, 68842, 69122, 69310, 69597, 70851, 71070, 71403, 71580, 72466, 73421, 73931, 74036, 74242, 76793, 77188, 77234, 77335, 78029, 78110, 78729, 78798, 78875, 81000, 81681, 83586, 84223, 85033, 85881, 86870, 87773, 93236, 102441, 120351, 121323, 129868, 141762, 1080265, 1584544]
OVERALL_CREW_COLS_ORDER = [1, 24, 36, 37, 42, 59, 108, 117, 122, 138, 149, 151, 153, 154, 172, 190, 224, 240, 280, 282, 309, 322, 339, 348, 356, 376, 384, 432, 434, 436, 465, 474, 488, 489, 491, 493, 494, 495, 508, 510, 525, 531, 541, 546, 547, 557, 561, 564, 578, 597, 598, 608, 636, 638, 664, 668, 673, 770, 793, 865, 869, 893, 894, 897, 898, 900, 908, 909, 947, 950, 997, 1012, 1032, 1035, 1044, 1046, 1060, 1071, 1091, 1093, 1095, 1113, 1150, 1152, 1188, 1213, 1221, 1223, 1224, 1225, 1243, 1255, 1259, 1262, 1263, 1264, 1296, 1301, 1302, 1307, 1325, 1357, 1461, 1484, 1524, 1527, 1528, 1530, 1551, 1593, 1617, 1650, 1720, 1723, 1729, 1760, 1776, 1884, 1927, 1938, 1999, 2031, 2043, 2073, 2212, 2215, 2216, 2226, 2236, 2238, 2240, 2242, 2260, 2289, 2294, 2303, 2324, 2484, 2507, 2523, 2532, 2636, 2702, 2704, 2710, 2723, 2725, 2862, 2874, 2949, 2950, 2952, 2997, 3026, 3027, 3146, 3175, 3192, 3193, 3275, 3311, 3317, 3358, 3388, 3393, 3501, 3535, 3556, 3562, 3658, 3686, 3687, 3769, 3776, 3806, 3831, 3893, 3953, 3965, 3987, 3989, 3996, 4014, 4023, 4061, 4140, 4185, 4222, 4350, 4387, 4401, 4406, 4415, 4429, 4500, 4501, 4504, 4507, 4700, 4723, 4767, 4952, 5026, 5140, 5144, 5162, 5216, 5281, 5288, 5328, 5359, 5362, 5363, 5381, 5398, 5488, 5489, 5490, 5491, 5493, 5553, 5572, 5575, 5602, 5628, 5634, 5656, 5666, 5669, 5912, 5914, 6041, 6044, 6046, 6048, 6111, 6159, 6189, 6210, 6346, 6347, 6348, 6377, 6410, 6468, 6593, 6648, 6818, 7020, 7068, 7182, 7187, 7200, 7229, 7232, 7262, 7395, 7396, 7413, 7418, 7494, 7537, 7623, 7624, 7714, 7727, 7728, 7735, 7779, 7800, 7879, 7885, 7903, 8246, 8303, 8376, 8844, 8846, 8858, 8885, 9027, 9039, 9040, 9062, 9152, 9178, 9181, 9196, 9204, 9217, 9248, 9251, 9349, 9545, 9573, 9587, 9619, 9789, 9966, 10440, 10494, 10572, 10766, 10771, 10781, 10828, 10830, 10850, 10965, 11091, 11092, 11098, 11099, 11181, 11269, 11371, 11401, 11409, 11472, 11475, 11505, 11614, 11649, 11770, 11874, 11905, 12235, 12453, 12833, 12987, 13166, 13223, 13563, 13585, 13588, 13848, 14139, 14351, 14377, 14536, 14639, 14712, 14764, 14765, 14999, 15005, 15017, 15189, 15221, 15344, 15347, 15426, 15493, 15524, 16177, 16294, 16300, 16363, 16425, 16483, 16736, 16830, 16938, 17016, 17146, 17209, 17210, 17211, 17282, 17428, 17698, 17825, 18311, 18457, 18897, 19155, 19292, 19303, 19310, 19689, 19971, 20228, 20229, 20540, 20629, 20821, 21036, 21678, 21962, 22061, 22815, 23486, 23545, 23787, 23966, 24190, 25236, 26175, 26760, 28615, 32035, 33008, 33009, 35073, 35694, 35736, 35771, 37281, 37710, 39123, 39996, 40383, 40384, 40810, 40813, 40823, 41039, 42267, 45829, 45830, 52042, 52161, 54211, 54419, 54734, 54777, 55710, 56765, 59287, 59839, 62778, 65429, 68016, 68602, 71273, 71797, 74978, 78747, 80602, 84348, 85670, 87550, 92376, 95901, 96627, 102429, 102445, 113073, 113194, 150975, 158916, 230436, 406204, 548445, 1007395, 1024910, 1034748, 1074163, 1077782, 1116937, 1172443, 1323090, 1324652, 1336716, 1338372, 1338976, 1340345, 1341403, 1341781, 1342657, 1342658, 1352966, 1352969, 1360097, 1367493, 1368867, 1371064, 1377220, 1378171, 1378240, 1378828, 1387183, 1393300, 1393558, 1394130, 1398972, 1399071, 1399141, 1400092, 1403415, 1404217, 1404244, 1424894, 1429549, 1447543, 1456696, 1548698, 1552521, 1733142, 1813644]
GENRE_COLS_ORDER = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']
ORIGINAL_LANGUAGES = ['ab', 'af', 'am', 'ar', 'ay', 'bg', 'bm', 'bn', 'bo', 'bs', 'ca', 'cn', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'gl', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'is', 'it', 'iu', 'ja', 'jv', 'ka', 'kk', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'nb', 'ne', 'nl', 'no', 'pa', 'pl', 'ps', 'pt', 'qu', 'ro', 'ru', 'rw', 'sh', 'si', 'sk', 'sl', 'sm', 'sq', 'sr', 'sv', 'ta', 'te', 'tg', 'th', 'tl', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xx', 'zh', 'zu', 'nan']

NUM_GENRES = len(GENRE_COLS_ORDER)
NUM_CAST = len(OVERALL_CAST_COLS_ORDER)
NUM_CREW = len(OVERALL_CREW_COLS_ORDER)
NUM_ORIGINAL_LANGUAGES = len(ORIGINAL_LANGUAGES)
MODEL_DIR = 'static/saved_models/'

def prepare_name_to_col_dict(df, cols_order):
    result = dict(zip(df['name'], df['id']))
    for name in result:
        result[name] = cols_order.index(result[name])
    return result

OVERALL_TOP_CAST = prepare_name_to_col_dict(pd.read_csv('static/top_cast.csv'), OVERALL_CAST_COLS_ORDER)
OVERALL_TOP_CREW = prepare_name_to_col_dict(pd.read_csv('static/top_crew.csv'), OVERALL_CREW_COLS_ORDER)
    
class OverallRevenuePredictorModel(nn.Module):
    def __init__(self, bert_embedding_size = 128, original_language_embedding_size = 32, cast_embedding_size = 32, crew_embedding_size = 32, hidden_size = 256):
        super(OverallRevenuePredictorModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Linear layer for textual embeddings
        self.linear_overview = nn.Linear(self.bert.config.hidden_size, bert_embedding_size)

        # Linear layer for original language embeddings
        self.linear_original_language = nn.Linear(NUM_ORIGINAL_LANGUAGES, original_language_embedding_size)

        # Linear layer for embedding cast
        self.linear_cast = nn.Linear(NUM_CAST, cast_embedding_size)

        # Linear layer for embedding crew
        self.linear_crew = nn.Linear(NUM_CREW, crew_embedding_size)

        # Budget and budget_unknown, and genres
        self.other_features_size = 2 + NUM_GENRES

        self.output_layer = nn.Sequential(
            nn.Linear(bert_embedding_size + original_language_embedding_size + cast_embedding_size + crew_embedding_size + self.other_features_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input):
        bert_out = self.bert(input_ids=input[:, :256].long(), attention_mask=input[:, 256:512].long())
        overview_embedding = self.linear_overview(bert_out['pooler_output'])
        overview_embedding = nn.LeakyReLU()(overview_embedding)

        original_language_embedding = self.linear_original_language(input[:, 512:512+NUM_ORIGINAL_LANGUAGES])
        original_language_embedding = nn.LeakyReLU()(original_language_embedding)
        cast_embedding = self.linear_cast(input[:, 512+NUM_ORIGINAL_LANGUAGES:512+NUM_ORIGINAL_LANGUAGES+NUM_CAST])
        cast_embedding = nn.LeakyReLU()(cast_embedding)
        crew_embedding = self.linear_crew(input[:, 512+NUM_ORIGINAL_LANGUAGES+NUM_CAST:512+NUM_ORIGINAL_LANGUAGES+NUM_CAST+NUM_CREW])
        cast_embedding = nn.LeakyReLU()(cast_embedding)
        other_features = input[:, 512+NUM_ORIGINAL_LANGUAGES+NUM_CAST+NUM_CREW:]


        return self.output_layer(torch.cat((
            overview_embedding,
            original_language_embedding,
            cast_embedding,
            crew_embedding,
            other_features
        ), dim=1))
        
class OverallRevenuePredictor():
    def __init__(self):
        self.model = OverallRevenuePredictorModel(
            bert_embedding_size = 192,
            original_language_embedding_size = 16,
            cast_embedding_size = 32,
            crew_embedding_size = 16,
            hidden_size = 192
        )
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model.load_state_dict(torch.load(
            MODEL_DIR + "overall_revenue_model_leakyReLU_hidden_192_bert_192_lang_16_cast_32_crew_16_batch_64.pth",
            map_location=torch.device('cpu')
        ))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_cast_tensor(self, cast_names):
        cast_tensor = torch.zeros(NUM_CAST, device=self.device)
        for name in cast_names:
            if name in OVERALL_TOP_CAST:
                cast_tensor[OVERALL_TOP_CAST[name]] = 1
        return cast_tensor

    def get_crew_tensor(self, crew_names):
        crew_tensor = torch.zeros(NUM_CREW, device=self.device)
        for name in crew_names:
            if name in OVERALL_TOP_CREW:
                crew_tensor[OVERALL_TOP_CREW[name]] = 1
        return crew_tensor


    def get_genre_tensor(self, genres):
        genre_tensor = torch.zeros(NUM_GENRES, device=self.device)
        for genre in genres:
            if genre in GENRE_COLS_ORDER:
                genre_tensor[GENRE_COLS_ORDER.index(genre)] = 1
        return genre_tensor
    
    def predict_revenue(self, movie_data):
        '''
        movie_data: list of dictionaries, each dictionary contains the following keys:
            - overview: string, overview of the movie, required
            - original_language: string, original language of the movie
            - cast: list of strings, names of the cast
            - crew: list of strings, names of the crew
            - genre_list: list of strings, genres of the movie
            - budget: int, budget of the movie (optional)
        '''
        encoded = self.tokenizer.encode_plus(
            movie_data['overview'], add_special_tokens=True,
            max_length=256, padding='max_length', truncation=True, return_tensors='pt'
        ).to(self.device)
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()

        genres = self.get_genre_tensor(movie_data['genre_list'] if 'genre_list' in movie_data else [])

        language_one_hot = [0] * NUM_ORIGINAL_LANGUAGES
        language_one_hot[ORIGINAL_LANGUAGES.index(movie_data['original_language'] if 'original_language' in movie_data else 'en')] = 1
        original_language = torch.tensor(language_one_hot, dtype=torch.float).to(self.device)

        cast = self.get_cast_tensor(movie_data['cast'] if 'cast' in movie_data else [])
        crew = self.get_crew_tensor(movie_data['crew'] if 'crew' in movie_data else [])

        if 'budget' in movie_data:
            budget = torch.tensor([movie_data['budget'] / 1e8], dtype=torch.float).to(self.device)
            budget_unknown = torch.tensor([0], dtype=torch.float).to(self.device)
        else:
            budget = torch.tensor([0], dtype=torch.float).to(self.device)
            budget_unknown = torch.tensor([1], dtype=torch.float).to(self.device)

        input = torch.cat((input_ids, attention_mask, original_language, cast, crew, budget, budget_unknown, genres), dim=0).unsqueeze(0)
        return self.model(input).item() * 1e8
    
if __name__ == '__main__':
    predictor = OverallRevenuePredictor()
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

    test_movie_actual_revenues = [1.1e9, 1.44e9, 140e6]
    for movie, actual_revenue in zip(test_movies, test_movie_actual_revenues):
        print(f"Predicting revenue for {movie['title']}")
        predicted_revenue = predictor.predict_revenue(movie)
        print(f"Predicted revenue: {predicted_revenue}")
        print(f"Actual revenue: {actual_revenue}")
        print()
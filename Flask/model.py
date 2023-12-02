import torch.nn as nn
import torch
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader

NUM_GENRES = 20
NUM_CAST = 0
NUM_CREW = 0
NUM_ORIGINAL_LANGUAGES = 90

# 3. Model
class RevenuePredictor(nn.Module):
    def __init__(self, bert_embedding_size = 128, original_language_embedding_size = 32, cast_embedding_size = 32, crew_embedding_size = 32, hidden_size = 256):
        super(RevenuePredictor, self).__init__()
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
    
class RevenueDataset(Dataset):
    def __init__(self, tokenizer, data, device, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data
        self.original_language_cols = [x for x in data.columns if x.startswith('original_language_')]
        self.genre_cols = [x for x in data.columns if x.startswith('genre_')]
        self.cast_cols = [x for x in data.columns if x.startswith('cast_')]
        self.crew_cols = [x for x in data.columns if x.startswith('crew_')]
        self.device = device

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = self.tokenizer.encode_plus(row['overview'], add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt').to(self.device)

        original_language = torch.tensor(row[self.original_language_cols].values.astype(float), dtype=torch.float, device=self.device)
        genres = torch.tensor(row[self.genre_cols].values.astype(float), dtype=torch.float, device=self.device)
        cast = torch.tensor(row[self.cast_cols].values.astype(float), dtype=torch.float, device=self.device)
        crew = torch.tensor(row[self.crew_cols].values.astype(float), dtype=torch.float, device=self.device)
        budget = torch.tensor(row['budget_100M'], dtype=torch.float, device=self.device)
        budget_unknown = torch.tensor(row['budget_unknown'], dtype=torch.float, device=self.device)
        revenue = torch.tensor(row['revenue_100M'], dtype=torch.float, device=self.device)

        x = torch.cat((
            inputs["input_ids"].squeeze(),
            inputs["attention_mask"].squeeze(),
            original_language,
            genres,
            cast,
            crew,
            budget.unsqueeze(0),
            budget_unknown.unsqueeze(0)
        ))

        return x, revenue

    def __len__(self):
      return len(self.data)
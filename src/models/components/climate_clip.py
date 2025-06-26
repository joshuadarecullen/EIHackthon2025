import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel, BertTokenizer

class ClimateDataEncoder(nn.Module):
    def __init__(self,
                 input_channels,
                 out_channels,
                 output_dim,
                 bert_freeze=False):

        super().__init__()


        self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.convpool = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.ModuleList([self.convpool for _ in range(2)])
        self.fc1 = nn.Linear(1200,output_dim)
        self.fc2 = nn.Linear(768, output_dim)

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


        self.temperature = nn.Parameter(torch.tensor(0.07))

        # Freeze BERT
        if bert_freeze == False:
            for param in self.bert_model.parameters():
                param.requires_grad = False

    def forward(self, climate_data, text):
        climate_features = self.encode_climate_data(climate_data)
        text_features = self.encode_text(text)
        return climate_features,text_features

    def encode_climate_data(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)
        for conv in self.conv2:
            x = conv(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.mean(dim=1)
        x = self.fc1(x)
        return x

    def encode_text(self, text):
        inputs = self.bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = self.bert_model(**inputs)
        pooled_output = outputs.pooler_output
        pooled_output = self.fc2(pooled_output)
        return pooled_output

    def loss_fn(self, climate_features, text_features, device):
        """
        Compute loss:
        Ground truth  acts as the target labels when comparing each image with the corresponding text. It assumes that:

            Metdata i is paired with Text i

        There is a one-to-one correspondence between the metdata and texts in each batch

        CLIP models compute a similarity matrix between all met_data and all texts in the batch.
        """
        text_embed = F.normalize(text_features, dim=-1)
        climate_embed = F.normalize(climate_features, dim=-1)
        logits = text_embed @ climate_embed.T  # dot product

        logits /= self.temperature.exp()
        labels = torch.arange(logits.size(0), device=device)

        loss_t2c = F.cross_entropy(logits, labels)         # text-to-climate
        loss_c2t = F.cross_entropy(logits.T, labels)       # climate-to-text

        loss = (loss_t2c + loss_c2t) / 2

        return loss

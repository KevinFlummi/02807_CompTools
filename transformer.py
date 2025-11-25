import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "lib")
)
from data_loading import make_splits

THIS_PATH = os.path.dirname(os.path.realpath(__file__))


class PairwiseRatingDataset(Dataset):
    """
    Wraps ReviewDataset.
    For each __getitem__, returns a random pair of review texts
    and a similarity label derived from the ratings.
    """

    def __init__(self, review_dataset):
        self.ds = review_dataset
        self.max_rating = max(r for _, r in self.ds)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Anchor review
        t1, r1 = self.ds[idx]

        # Random second review
        rand_idx = random.randint(0, len(self.ds) - 1)
        t2, r2 = self.ds[rand_idx]

        # Rating distance to similarity in [0,1]
        sim = 1.0 - abs(r1 - r2) / self.max_rating
        sim = torch.tensor(sim, dtype=torch.float32)

        return t1, t2, sim


class MeanPooling(nn.Module):
    """
    Averages token embeddings using the attention mask
    """

    def forward(self, token_embeddings, attention_mask):
        # expand mask to correct dimensions and apply mask to embeddings
        mask = attention_mask.unsqueeze(-1).float()
        masked = token_embeddings * mask
        # sum over masked values and divide by count to get avg
        summed = masked.sum(dim=1)
        counted = mask.sum(dim=1)
        return summed / counted


class BasicSentenceTransformer(nn.Module):
    """
    Very basic transformer model using just an encoder and mean pooling
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.pool = MeanPooling()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        return self.pool(token_embeddings, attention_mask)


class ContrastiveLoss(nn.Module):
    """
    Loss function using contrastive cosine similarity;
    uses sigmoid to push high similarity for positives towards 1 and low similarity towards -1
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, emb1, emb2, labels):
        # get cosine sim
        sim = self.cos(emb1, emb2) / self.temperature
        # get positive pair loss (max sim)
        pos = -labels * torch.log(torch.sigmoid(sim) + 1e-8)
        # get negative pair loss (min sim)
        neg = -(1 - labels) * torch.log(1 - torch.sigmoid(sim) + 1e-8)
        # return mean of both
        return (pos + neg).mean()


def build_transformer():
    # start from untrained, small bert model
    config = AutoConfig.from_pretrained(
        "bert-base-uncased",
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
    )
    encoder = AutoModel.from_config(config)
    return BasicSentenceTransformer(encoder)


def collate_batch(batch, tokenizer):
    """
    Tokenizes two parallel lists of texts and stacks their similarity labels
    """
    texts1 = [b[0] for b in batch]
    texts2 = [b[1] for b in batch]
    labels = torch.stack([b[2] for b in batch])

    enc1 = tokenizer(texts1, return_tensors="pt", padding=True, truncation=True)
    enc2 = tokenizer(texts2, return_tensors="pt", padding=True, truncation=True)

    # remove token_type_ids to prevent error during model.forward() (because we use very basic model)
    if "token_type_ids" in enc1:
        del enc1["token_type_ids"]
        del enc2["token_type_ids"]

    return enc1, enc2, labels


def train_sentence_transformer(train_ds, epochs=1, batch_size=16, lr=2e-5):
    """
    Training function for transformer
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = build_transformer()

    # we need the pairwise dataset to train on similarity stuff
    pairwise_dataset = PairwiseRatingDataset(train_ds)

    dataloader = DataLoader(
        pairwise_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, tokenizer),
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = ContrastiveLoss()

    for ep in range(epochs):
        print(f"Epoch {ep + 1}/{epochs}")
        for step, (enc1, enc2, labels) in enumerate(dataloader):
            # encode both texts individually
            emb1 = model(**enc1)
            emb2 = model(**enc2)

            # get loss val from embeddings
            loss = loss_fn(emb1, emb2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"  Step {step}, Loss = {loss.item():.4f}")

    return model, tokenizer


def encode_texts(model, tokenizer, texts, batch_size=64):
    """
    Encodes list of texts into embeddings using model and tokenizer
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # remove unused fields
        enc.pop("token_type_ids", None)
        enc.pop("position_ids", None)

        with torch.no_grad():
            emb = model(**enc)

        all_embeddings.append(emb)
    # return values as numpy array
    return torch.cat(all_embeddings, dim=0).cpu().numpy()


if __name__ == "__main__":
    train_ds, val_ds, test_ds = make_splits(
        os.path.join(THIS_PATH, "datasets", "Handmade_Products_f.jsonl")
    )
    # we only use this as a demo, so we use a relatively small ds (just 1 category) and 1 epoch should be enough (also it takes forever to train)
    model, tokenizer = train_sentence_transformer(train_ds, epochs=1)
    torch.save(model.state_dict(), "sentence_tranformer_weights.pth")
    tokenizer.save_pretrained("tokenizer/")

from transformers import BertTokenizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.DataFrame({
    'text': [
        "I love this movie, it's amazing!",
        "This film was terrible and boring.",
        "The plot was okay but not great."
    ],
    'label': ['positive', 'negative', 'neutral']
})

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(texts, max_length=128):
    encodings = tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'token_type_ids': encodings['token_type_ids']
    }

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['label'])
processed_data = preprocess_text(data['text'])

print("Input IDs shape:", processed_data['input_ids'].shape)
print("Attention Mask shape:", processed_data['attention_mask'].shape)
print("Encoded Labels:", encoded_labels)

np.save('input_ids.npy', processed_data['input_ids'])
np.save('attention_mask.npy', processed_data['attention_mask'])
np.save('labels.npy', encoded_labels)

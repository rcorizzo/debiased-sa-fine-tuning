import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
# physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
# sys.exit(1)
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow.keras.backend as K
import tensorflow.keras.backend as kb
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from tqdm import tqdm

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def prepare_data(input_text_list, tokenizer):
    tokens = tokenizer.batch_encode_plus(
        input_text_list,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(tokens['input_ids'], tf.float64),
        'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)
    }


def sentiment_scores(input_text_list, model):
    processed_data = prepare_data(input_text_list, tokenizer)
    probs = model.predict(processed_data)
    return probs

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

models = {}
ethnicities = ["Celtic", "European", "African", "Muslim", "EastAsian", "Hispanic", "SouthAsian", "Nordic"]
ethnicities = ["Celtic"]
ethn = "Celtic"
path = "../models/sentiment_model_frozen_layers_10_epochs_both_"

model = tf.keras.models.load_model(path + ethn)

comments_sample = pd.read_csv("../input/all_classifications.csv")
predicted_stars = {}

predicted_stars[ethn] = []

batch_size = 2400  # Number of comments to process in each batch
print('starting loop')
for i in tqdm(range(0, len(comments_sample), batch_size)):
    batch_comments = comments_sample['comment'].iloc[i:i+batch_size].tolist()
    probs = sentiment_scores(batch_comments, model)
    predicted_stars[ethn].extend(probs)


predicted_stars[ethn] = np.argmax(np.array(predicted_stars[ethn]), axis=1) + 1
comments_sample["predicted_stars_" + ethn] = predicted_stars[ethn]


comments_sample.to_csv(f"../output/comments_equal_sample_equal_stars_{ethn}_batched.csv", index=False)
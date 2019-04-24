local output_dim = 768;
local max_seq_len = 100
{
  "dataset_reader": {
    "type": "sst_tokens",
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "max_pieces": max_seq_len,
        "do_lowercase": true     
      }
    }
  },
  "train_data_path": "data/stanfordSentimentTreebank/trees/train.txt",
  "validation_data_path": "data/stanfordSentimentTreebank/trees/dev.txt",

  "model": {
    "type": "lstm_classifier",
    "word_embeddings": {
      "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": "bert-base-uncased"
        },
      "allow_unmatched_keys":true
      },
    "encoder": {
      "type": "bert",
      "output_dim": output_dim
     }
   },
      "iterator": {
        "type": "bucket",
        "batch_size": 32,
        "sorting_keys": [["tokens", "num_tokens"]]
      },
      "trainer": {
        "optimizer": "adam",
        "patience": 5
      }
  
}


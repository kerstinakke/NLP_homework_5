local bert_dim = 768;
local max_seq_len = 100;
local learning_rate = 0.001;
local weight_decay = 0.005;

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
      "type": "transformer-seq2vec",
      "input_dim": bert_dim,
      "hidden_dim": 256,
      "projection_dim": 128,
      "feedforward_hidden_dim": 128,
      "num_attention_heads": 8
     }
   },
      "iterator": {
        "type": "bucket",
        "batch_size": 32,
        "sorting_keys": [["tokens", "num_tokens"]]
      },
      "trainer": {
        "optimizer": {
            "type":"adam",
            "lr": learning_rate,
            "weight_decay": weight_decay
            },
        "patience": 5
         
      }
  
}


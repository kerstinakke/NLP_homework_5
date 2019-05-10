from typing import List

from overrides import overrides
import torch
from torch.nn import Dropout
import numpy as np

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder
from allennlp.nn import util

@Seq2VecEncoder.register("transformer-seq2vec")
class TransformerSeq2VecEncoder(Seq2VecEncoder):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 projection_dim: int,
                 feedforward_hidden_dim: int,
                 num_attention_heads: int,
                 num_layers: int = 1,
                 use_positional_encoding: bool = True,
                 dropout_prob: float = 0.1,
                 residual_dropout_prob: float = 0.2,
                 attention_dropout_prob: float = 0.1) -> None:
        super(TransformerSeq2VecEncoder, self).__init__()
        self.seq2seq = StackedSelfAttentionEncoder(input_dim,hidden_dim,projection_dim,feedforward_hidden_dim,num_layers,
                                                  num_attention_heads,use_positional_encoding,dropout_prob
                                                   ,residual_dropout_prob,attention_dropout_prob)
        
    @overrides
    def get_input_dim(self) -> int:
        return self.seq2seq._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.seq2seq._output_dim

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
        return torch.sum(self.seq2seq(inputs, torch.tensor(np.ones(inputs.shape[:2]))) , dim = 1)
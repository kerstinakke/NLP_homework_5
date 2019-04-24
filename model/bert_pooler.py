from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
import torch
from overrides import overrides

@Seq2VecEncoder.register("bert")
class BertSentencePooler(Seq2VecEncoder):
    def __init__(self,output_dim):
        super().__init__() 
        self.output_dim = output_dim
        
    def forward(self, embs: torch.tensor, 
                mask: torch.tensor=None) -> torch.tensor:
        # extract first token tensor
        return embs[:, 0]
     
    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim 
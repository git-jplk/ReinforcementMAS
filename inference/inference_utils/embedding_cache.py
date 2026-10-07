from git import List
import torch


class EmbeddingCache():
    def __init__(self, replace :bool = True):
        self.replace = replace
        self.cache: List[torch.Tensor] = []
        
    def retrieve(self) -> List[torch.Tensor]:
        return self.cache
    def load(self, embeddings: List[torch.Tensor]) -> None:
        if self.replace:
            self.cache = embeddings
        else:
            self.cache.extend(embeddings)
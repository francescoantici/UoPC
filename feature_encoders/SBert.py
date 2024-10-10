from sentence_transformers import SentenceTransformer

class SBert:
        
    def __init__(self, weights = "all-MiniLM-L6-v2") -> None:
       self._encoder = SentenceTransformer(weights)
       
    def encode_job(self, x:list) -> list:
        try:
            return self._encoder.encode(self._parse_data(x)) 
        except Exception as e:
            print(e)
            return []
    
    def _parse_job_data(self, features:list):
        return ",".join(map(str, features))    

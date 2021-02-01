from transformers import AutoTokenizer, AutoModel
import time

class SPECTERizer():
  
  def __init__(self):
    self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    self.model = AutoModel.from_pretrained("allenai/specter")
    
  def epoch_time(start_time, end_time):
    """ Calculate time for inference
    """
    time_elapsed = end_time - start_time
    epoch_mins = int(time_elapsed / 60)
    epoch_secs = time_elapsed - (epoch_mins * 60)
    return epoch_mins, epoch_secs
    
  def generate_embedding(input):
    """ This method will enable us to generate embedding using input from user
    """
    input_ids = self.tokenizer(input, padding=True, truncation=True, return_tensors="pt", max_length=512)
    embedding = self.model(**input_ids).last_hidden_state[:, 0, :]
    return embedding

  
  



from word_dictionary import WordDictionary
from torch.utils.data import Dataset
from copy import copy

class MyDataset(Dataset):
  def __init__(self, word_dict: WordDictionary, mode: str) -> None:
    super().__init__()
    self.mode = mode
    
    self.word_dict = word_dict
    self.src_idlines_train = word_dict.get_id("train-1.short", "ja")
    self.src_idlines_dev = word_dict.get_id("dev", "ja")
    self.src_idlines_test = word_dict.get_id("test", "ja")
    self.dst_idlines_train = word_dict.get_id("train-1.short", "en")
    self.dst_idlines_dev = word_dict.get_id("dev", "en")
    self.dst_idlines_test = word_dict.get_id("test", "en")
    self.special_token = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3}
  
  def __len__(self):
    if self.mode == "train":
      return len(self.src_idlines_train)
    elif self.mode == "dev":
      return len(self.src_idlines_dev)
    else:
      return len(self.dst_idlines_test)
    
  def __getitem__(self, idx):
    if self.mode == "train":
      return self.__add_bos_eos(self.src_idlines_train[idx]), self.__add_bos_eos(self.dst_idlines_train[idx])
    elif self.mode == "dev":
      return self.__add_bos_eos(self.src_idlines_dev[idx]), self.__add_bos_eos(self.dst_idlines_dev[idx])
    else:
      return self.__add_bos_eos(self.src_idlines_test[idx]), self.__add_bos_eos(self.dst_idlines_test[idx])
  
  def get_vocab_size(self):
    return len(self.word_dict.get_dict("ja", "w2id")), len(self.word_dict.get_dict("en", "w2id"))
  
  def __add_bos_eos(self, ids):
    tmp_list = copy(ids)
    tmp_list.insert(0, self.special_token["<bos>"])
    tmp_list.append(self.special_token["<eos>"])
    return tmp_list
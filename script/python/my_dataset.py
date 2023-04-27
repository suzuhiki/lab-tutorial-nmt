from id_store import IdStore
from torch.utils.data import Dataset

class MyDataset(Dataset):
  def __init__(self, id_store: IdStore, mode: str) -> None:
    super().__init__()
    self.src_lang = "ja"
    self.dst_lang = "en"
    self.mode = mode
    
    self.id_store = id_store
    self.src_idlines_train = id_store.get_id("train", self.src_lang)
    self.src_idlines_dev = id_store.get_id("dev", self.src_lang)
    self.src_idlines_test = id_store.get_id("test", self.src_lang)
    self.dst_idlines_train = id_store.get_id("train", self.dst_lang)
    self.dst_idlines_dev = id_store.get_id("dev", self.dst_lang)
    self.dst_idlines_test = id_store.get_id("test", self.dst_lang)
    self.special_token = {"<PAD>":0, "<BOS>":1, "<EOS>":2, "<UNK>":3}
  
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
    dict = self.id_store.get_dict()
    return len(dict[0]), len(dict[2])
  
  def __add_bos_eos(self, ids):
    tmp_list = ids
    tmp_list.insert(0, self.special_token["<BOS>"])
    tmp_list.append(self.special_token["<EOS>"])
    return ids
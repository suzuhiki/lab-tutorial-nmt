from id_attacher import IdAttacher
import pickle

class IdStore:
  def __init__(self) -> None:
    self.SRC_PATH = "/home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/"
    self.SRC_EN_NAMES = ["train-1.short.en", "dev.en", "test.en"]
    self.SRC_JA_NAMES = ["train-1.short.ja", "dev.ja", "test.ja"]
    self.load_id_dic = {"train":"train-1.short.", "dev":"dev.", "test":"test."}
    self.DATA_PATH = "/home/morioka/workspace/git_projects/lab-tutorial-nmt/data/"
    self.id_attacher_en = IdAttacher()
    self.id_attacher_ja = IdAttacher()
  
  def create_id(self):
    # create EN dict
    with open(self.SRC_PATH + "train-1.short.en", "r") as f:
      tmp_lines = f.read().splitlines()
      tmp_doc = list(map(lambda line: line.split(" "),tmp_lines))
      self.id_attacher_en.add_dict(tmp_doc)
      self.id_attacher_en.save_dict("en")
      print("id -> word")
      print(self.id_attacher_en.id2w_dict)
      print(len(self.id_attacher_en.id2w_dict))
      print("word -> id")
      print(self.id_attacher_en.w2id_dict)
      print(len(self.id_attacher_en.w2id_dict))

    # convert EN srcs
    for name in self.SRC_EN_NAMES:
      src_idlines = []
      with open(self.SRC_PATH + name, "r") as f:
        src_lines = f.read().splitlines()
        for line in src_lines:
          tmp_wordlist = line.split(" ")
          tmp_wordlist = list(map(lambda word: self.id_attacher_en.word2id(word), tmp_wordlist))
          src_idlines.append(tmp_wordlist)
      with open(self.DATA_PATH + name + ".bin", "wb") as f:
        pickle.dump(src_idlines, f)
      print(f"convert completed:{name}")
      print(src_idlines)

    # create JA dict
    with open(self.SRC_PATH + "train-1.short.ja", "r") as f:
      tmp_lines = f.read().splitlines()
      tmp_doc = list(map(lambda line: line.split(" "),tmp_lines))
      self.id_attacher_ja.add_dict(tmp_doc)
      self.id_attacher_ja.save_dict("ja")
      print("id -> word")
      print(self.id_attacher_ja.id2w_dict)
      print(len(self.id_attacher_ja.id2w_dict))
      print("word -> id")
      print(self.id_attacher_ja.w2id_dict)
      print(len(self.id_attacher_ja.w2id_dict))

    # convert JA srcs
    for name in self.SRC_JA_NAMES:
      src_idlines = []
      with open(self.SRC_PATH + name, "r") as f:
        src_lines = f.read().splitlines()
        for line in src_lines:
          tmp_wordlist = line.split(" ")
          tmp_wordlist = list(map(lambda word: self.id_attacher_ja.word2id(word), tmp_wordlist))
          src_idlines.append(tmp_wordlist)
      with open(self.DATA_PATH + name + ".bin", "wb") as f:
        pickle.dump(src_idlines, f)
      print(f"convert completed:{name}")
      print(src_idlines)
  
  def get_id(self, data_id: str, lang: str):
    if data_id in self.load_id_dic:
      if lang in ["en", "ja"]:
        path = self.DATA_PATH + self.load_id_dic[data_id] + lang + ".bin"
        with open(path, "rb") as f:
          return pickle.load(f)
  
  def get_dict(self):
    return [self.id_attacher_en.load_dict("en"), self.id_attacher_ja.load_dict("ja")]
        
  def print_dict(self):
    dict = self.get_dict()
    print(dict)




from word_dictionary import IdAttacher
import pickle

def convert_to_id():
  SRC_PATH = "/home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/"
  SRC_EN_NAMES = ["train-1.short.en", "dev.en", "test.en"]
  SRC_JA_NAMES = ["train-1.short.ja", "dev.ja", "test.ja"]
  DATA_PATH = "/home/morioka/workspace/git_projects/lab-tutorial-nmt/data/"
  
  # create EN dict
  id_attacher_en = IdAttacher()
  with open(SRC_PATH + "train-1.short.en", "r") as f:
    tmp_lines = f.read().splitlines()
    tmp_doc = list(map(lambda line: line.split(" "),tmp_lines))
    id_attacher_en.add_dict(tmp_doc)
    print("id -> word")
    print(id_attacher_en.id2w_dict)
    print("word -> id")
    print(id_attacher_en.w2id_dict)
  
  # convert EN srcs
  for name in SRC_EN_NAMES:
    src_idlines = []
    with open(SRC_PATH + name, "r") as f:
      src_lines = f.read().splitlines()
      for line in src_lines:
        tmp_wordlist = line.split(" ")
        tmp_wordlist = list(map(lambda word: id_attacher_en.word2id(word), tmp_wordlist))
        src_idlines.append(tmp_wordlist)
    with open(DATA_PATH + name + ".bin", "wb") as f:
      pickle.dump(src_idlines, f)
    print(f"convert completed:{name}")
    print(src_idlines)
    
  # create JA dict
  id_attacher_ja = IdAttacher()
  with open(SRC_PATH + "train-1.short.ja", "r") as f:
    tmp_lines = f.read().splitlines()
    tmp_doc = list(map(lambda line: line.split(" "),tmp_lines))
    id_attacher_ja.add_dict(tmp_doc)
    print("id -> word")
    print(id_attacher_ja.id2w_dict)
    print("word -> id")
    print(id_attacher_ja.w2id_dict)
  
  # convert JA srcs
  for name in SRC_JA_NAMES:
    src_idlines = []
    with open(SRC_PATH + name, "r") as f:
      src_lines = f.read().splitlines()
      for line in src_lines:
        tmp_wordlist = line.split(" ")
        tmp_wordlist = list(map(lambda word: id_attacher_ja.word2id(word), tmp_wordlist))
        src_idlines.append(tmp_wordlist)
    with open(DATA_PATH + name + ".bin", "wb") as f:
      pickle.dump(src_idlines, f)
    print(f"convert completed:{name}")
    print(src_idlines)




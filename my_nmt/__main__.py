from .other.word_dictionary import WordDictionary
from .util.util import get_swap_dict

if __name__ == "__main__":
    src = "/home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/train-1.short.en"
    word_dict = WordDictionary()
    print(word_dict.create_get_dict(src)[1])
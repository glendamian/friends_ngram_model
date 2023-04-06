# imports go here
import sys
# feel free to add imports.
from collections import Counter
import numpy as np

"""
Don't forget to put your name and a file comment here.

CS 4120
HW 2
Name: Glen Damian Lim

This is an individual homework.
"""


# Feel free to implement more helper functions
"""
Provided helper functions
"""
def read_sentences(filepath):
  """
  Reads contents of a file line by line.
  Parameters:
    filepath (str): file to read from
  Return:
    list of strings
  """
  f = open(filepath, "r")
  sentences = f.readlines()
  f.close()
  return sentences

def get_data_by_character(filepath):
  """
  Reads contents of a script file line by line and sorts into 
  buckets based on speaker name.
  Parameters:
    filepath (str): file to read from
  Return:
    dict of strings to list of strings, the dialogue that speaker speaks
  """
  char_data = {}
  script_file = open(filepath, "r", encoding="utf-8")
  for line in script_file:
    # extract the part between <speaker> tags
    speakers = line[line.index("<speakers>") + len("<speakers>"): line.index("</speakers>")].strip()
    if not speakers in char_data:
      char_data[speakers] = []
    char_data[speakers].append(line)
  return char_data

"""
This is your Language Model class
"""

class LanguageModel:
  # constants to define pseudo-word tokens
  # access via self.UNK, for instance
  UNK = "<UNK>"

  def __init__(self, n_gram, is_laplace_smoothing, line_begin = "<line>", line_end="</line>"):
    """Initializes an untrained LanguageModel
    Parameters:
      n_gram (int): the n-gram order of the language model to create
      is_laplace_smoothing (bool): whether or not to use Laplace smoothing
      line_begin (str): the token designating the beginning of a line
      line_end (str): the token designating the end of a line
    """
    self.line_begin = line_begin
    self.line_end = line_end
    # your other code here
    self.is_laplace_smoothing = is_laplace_smoothing
    self.n_gram = n_gram
    # tokens in current model
    self.tokens = []
    # vocabulary of current model
    self.vocab = {}
    # generated n_grams for current model
    self.n_grams = []
    # total occurences of all n-grams
    self.total_occ = {}
    # total occurences of all n-1 grams
    self.minus_one_occ = {}
    # trained data for current model
    self.trained_data = {}
    
  def make_ngrams(self, tokens, n):
      """Creates ngrams for the given sentences
      Parameters:
        sentence (list): list of tokens as strings from the training file
        
      Returns:
        list: list of tuples of strings, each tuple will represent an individual n-grams
      """
      n_grams = [tokens[i:(i+n)] for i in range(len(tokens) - n + 1)]
      return n_grams


  def mle(self, curr_n_gram, total_occ, minus_one_occ):
    """Calculate the MLE for the current input n-gram
    Parameters:
      curr_n_gram (tuples): tuples of string, representing our current n_gram
      total_occ (Counter): a counter dictionary containing the occurences of all n-grams
      minus_one_occ (Counter): a counter dictionary containing the occurences of all n-1 grams

    Returns:
      float: the MLE value of the current n-gram
    """
    # total tokens (N)
    tokens_count = len(self.tokens)
    # vocab size (|V|)
    vocab_size = len(self.vocab)
    # total occurences of current ngrams
    curr_count = total_occ[str(curr_n_gram)]
    # total occurences of current n-1 grams
    minus_one_count = minus_one_occ[str(curr_n_gram[:-1])]
    # calculation for unigram
    if self.n_gram == 1:
      if self.is_laplace_smoothing:
        return (curr_count + 1)/(tokens_count + vocab_size)
      else:
        return curr_count/ tokens_count

    # for any n-grams other than unigrams
    if self.is_laplace_smoothing:
      return (curr_count + 1)/(minus_one_count + vocab_size)
    else:
       return curr_count/minus_one_count

  def train(self, sentences):
    """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with line_begin and end with line_end
    Parameters:
      sentences (list): list of strings, one string per line in the training file

    Returns:
    None
    """
    # model tokens and vocabulary
    self.tokens = " ".join(sentences).split()
    self.vocab = Counter(self.tokens)
    # replace token that occurs once with unknown tokens
    self.tokens = [token.replace(token,self.UNK) if self.vocab[token] == 1 else token for token in self.tokens]
    # replace vocab that occurs once with unknown tokens
    self.vocab = Counter(self.tokens)

    trained_data = Counter()
    self.n_grams = self.make_ngrams(self.tokens, self.n_gram)
    minus_one_n_grams = self.make_ngrams(self.tokens, self.n_gram - 1)

    # total occurences of all n-grams
    self.total_occ = Counter(str(n_gram) for n_gram in self.n_grams)
    # total occurences of all n-1 grams
    self.minus_one_occ = Counter(str(n_gram) for n_gram in minus_one_n_grams)
    # iterate through model n-grams to calculate probability
    for n_gram in self.n_grams:
      trained_data[tuple(n_gram)] = self.mle(n_gram, self.total_occ, self.minus_one_occ)
    self.trained_data = trained_data
      

  def score(self, sentence):
    """Calculates the probability score for a given string representing a single sentence.
    Parameters:
      sentence (str): a sentence with tokens separated by whitespace to calculate the score of
      
    Returns:
      float: the probability value of the given string for this model
    """
    # current sentence tokens (replace with unknown token if word is not in vocabulary) and n-grams
    sentence_tokens = sentence.split()
    sentence_tokens = [token.replace(token,self.UNK) if token not in self.vocab else token for token in sentence_tokens]
    sentence_n_grams = self.make_ngrams(sentence_tokens, self.n_gram)
    # if probability is not zero we obtain, else we recalculate mle for the current unseen n-gram
    probs = [self.trained_data[tuple(n_gram)] if self.trained_data[tuple(n_gram)] != 0 else self.mle(n_gram, self.total_occ, self.minus_one_occ) 
    for n_gram in sentence_n_grams]
    # chain rule to multiply all probabilities
    return np.exp(np.sum(np.log(probs)))

  def get_next_word(self, curr_token):
    """Gets the next word for sentence generation.
    Parameters:
      curr_token (tuples): tuples of string, representing n-grams consisting our last n-1 tokens in our generated sentence so far
      
    Returns:
      string: the next word for our generated sentence
    """
    # finding possible n-grams
    possible_n_grams = {}
    if self.n_gram == 1:
      possible_n_grams = self.trained_data
    else:
      for k,v in self.trained_data.items():
        # find list of n-grams that has a prefix matching our current n-gram
        n_minus_one_k = list(k[:-1])
        if n_minus_one_k == curr_token:
          possible_n_grams[k] = v

    possible_words = list(possible_n_grams.keys())
    probs = list(possible_n_grams.values())
    # normalize probabilities and get index
    random_choice = np.random.choice(len(possible_words),p = probs / np.sum(probs))
    # get the last word from the selected n-gram as our next word
    next_word = possible_words[random_choice][-1]
    return next_word

  def generate_sentence(self):
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      string: the generated sentence
    """
    # initialize resulting sentence with line begin token depending on the number of n
    if self.n_gram == 1:
      sentence = [self.line_begin]
    else:
      sentence = [self.line_begin] * (self.n_gram - 1)

    next_word = ""
    # while last n tokens are not sentence end
    while next_word != self.line_end:
      curr_token = sentence[-(self.n_gram - 1):]
      next_word = self.get_next_word(curr_token)
      sentence.append(next_word)
    # for n > 2 case, we n - 2 append line ends at end of the sentence, as we have appended one in the previous while loop.
    # In total, we will have n - 1 line ends.
    if self.n_gram > 2:
        for _ in range(self.n_gram - 2):
          sentence.append(self.line_end)
    return ' '.join(sentence)

  def generate(self, n):
    """Generates n sentences from a trained language model using the Shannon technique.
    Parameters:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing strings, one per generated sentence
    """
    sentences = []
    if self.n_gram == 1:
      self.trained_data.pop(tuple([self.line_begin]))
    while n > 0:
      sentences.append(self.generate_sentence())
      n -= 1
    return sentences
  
  def find_best_match(self, character, test):
    """Finds the linguistically most similar character (has the highest average score) to the given character
    Parameters: 
      character (string): the given/current character
      test (dict) : a map between a character's name (string) and its sentences from the testing data (list)

    Returns: 
      tuples: the best matching character as string and its score as float
    """ 
    average_scores = {}
    for k,v in test.items():
      if k != character:
        average_scores[k] = np.mean([self.score(sentence) for sentence in v])
    best_match, best_score = max(average_scores.items(), key = lambda score: score[1])
    return best_match, best_score

  def perplexity(self, test_sequence):
    """Measures the perplexity for the given test sequence with this trained model. 
    Parameters: 
      test_sequence (string): a sequence of space-separated tokens to measure the perplexity of
    Returns: 
      float: the perplexity of the given sequence
    """ 
    probs = self.score(test_sequence)
    N = len([token for token in test_sequence.split() if token != self.line_begin])
    perplexity = np.power((1 / probs), 1 / N)
    return perplexity
  
def main():
  # TODO: implement the rest of this!
  ngram = int(sys.argv[1])
  training_path = sys.argv[2]
  testing_path = sys.argv[3]
  line_begin = sys.argv[4]
  if len(sys.argv) == 5:
    print("Runnning for", ngram, "model")
    print("\n")
    # instantiate a language model like....
    ngram_lm = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
    # training and testing sets
    train = read_sentences(training_path)
    test = read_sentences(testing_path)
    # training language model
    ngram_lm.train(train)
    # generating 25 sentences using trained model
    generated = ngram_lm.generate(25)
    [print(sentence + "\n") for sentence in generated]
    # average score for sentences in testing set
    average = [ngram_lm.score(sentence) for sentence in test]
    # first five sentences from testing set
    first_five = " ".join(test[:5])
    # calculate perplexity
    perplexity = ngram_lm.perplexity(first_five)
    # number of sentences in testing set
    print("Number of test sentences:", len(test))
    print("Average probability score:", np.mean(average))
    print("Standard deviation:", np.std(average))
    print("Perplexity for", str(ngram) + "-grams :", perplexity)
    print("\n")
    print("First 5 sentences:", first_five)

  else:
    train = read_sentences(training_path)
    test = read_sentences(testing_path)
    # Training data per character (list of sentences)
    train_monica = [sentence for sentence in train if "<speakers> Monica Geller </speakers>" in sentence]
    train_ross = [sentence for sentence in train if "<speakers> Ross Geller </speakers>" in sentence]
    train_chandler = [sentence for sentence in train if "<speakers> Chandler Bing </speakers>" in sentence]
    train_joey = [sentence for sentence in train if "<speakers> Joey Tribbiani </speakers>" in sentence]
    train_rachel = [sentence for sentence in train if "<speakers> Rachel Green </speakers>" in sentence]
    train_phoebe = [sentence for sentence in train if "<speakers> Phoebe Buffay </speakers>" in sentence]
    # Testing data per character (list of sentences)
    test_data = {"Monica Geller" : [sentence for sentence in test if "<speakers> Monica Geller </speakers>" in sentence], "Ross Geller" : [sentence for sentence in test if "<speakers> Ross Geller </speakers>" in sentence], 
    "Chandler Bing": [sentence for sentence in test if "<speakers> Chandler Bing </speakers>" in sentence], "Joey Tribbiani": [sentence for sentence in test if "<speakers> Joey Tribbiani </speakers>" in sentence], 
    "Rachel Green": [sentence for sentence in test if "<speakers> Rachel Green </speakers>" in sentence], "Phoebe Buffay": [sentence for sentence in test if "<speakers> Phoebe Buffay </speakers>" in sentence],}
    # # Language models per character
    lm_monica = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
    lm_ross = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
    lm_chandler = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
    lm_joey = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
    lm_rachel = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
    lm_phoebe = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
    # # Train each language model corresponding to each character's dataset
    lm_monica.train(train_monica)
    lm_ross.train(train_ross)
    lm_chandler.train(train_chandler)
    lm_joey.train(train_joey)
    lm_rachel.train(train_rachel)
    lm_phoebe.train(train_phoebe)
    # Test each language model corresponding to each character's dataset
    characters = {"Monica Geller" : lm_monica, "Ross Geller" : lm_ross, "Chandler Bing" : lm_chandler, "Joey Tribbiani" : lm_joey, "Rachel Green": lm_rachel, "Phoebe Buffay" :lm_phoebe}
    for k,v in characters.items():
      # find the best matching character
      best_match = v.find_best_match(k, test_data)
      print("Comparing:", k)
      print("Best match:", best_match[0])
      print("With average score:", best_match[1])
      print("\n")
    
if __name__ == '__main__':
    
  # make sure that they've passed the correct number of command line arguments
  if len(sys.argv) < 5:
    print("Usage:", "python lm.py ngram training_file.txt testingfile.txt line_begin [character]")
    sys.exit(1)

  main()


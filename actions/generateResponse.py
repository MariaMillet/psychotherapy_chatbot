import datasets
from datasets import load_dataset
from datasets import DatasetDict
from datasets.load import load_from_disk

def generate_N_grams(text,ngram=1):
  words=[word for word in text.split(" ")] 
  # print("Sentence after removing stopwords:",words)
  temp=zip(*[words[i:] for i in range(0,ngram)])
  ans=[' '.join(ngram) for ngram in temp]
  return ans    

def generate_n_gram(sentence, max_n_gram):
  n_gram_list = [generate_N_grams(sentence, n_gram) for n_gram in range(1, max_n_gram+1)]
  return n_gram_list

def generate_weights(coefficient_increase, number_of_weights):
  coefficients = [coefficient_increase**i for i in range(number_of_weights)]
  first_weight = 1 / sum(coefficients) 
  weights = [first_weight*coefficient_increase**i for i in range(number_of_weights)]
  return weights

def novelty_score(sent_1_n_grams, sent_2_n_grams):
#   print(sent_1_n_grams, sent_2_n_grams)
  common_n_grams = [set(sent_1_n_grams[i]).intersection(set(sent_2_n_grams[i])) for i in range(len(sent_1_n_grams))]
  weights = generate_weights(coefficient_increase=1.1, number_of_weights=len(sent_1_n_grams))
  try:
    scores = [len(common_n_grams[i])/min(len(sent_1_n_grams[i]), len(sent_2_n_grams[i])) for i in range(len(common_n_grams))]
    weighted_scores = [weight*score for weight, score in zip(weights, scores)]
  except ZeroDivisionError:
    weighted_scores = [0]

  return sum(weighted_scores)

def calculate_novelty(sentence,past_sentences):
  new_sentence_n_grams = generate_n_gram(sentence, max_n_gram=3)
#   print(len(past_sentences))
  similarity = sum([novelty_score(new_sentence_n_grams, past_sentences[str(i)]) for i in range(len(past_sentences))])/len(past_sentences)
  novelty = 1 - similarity
  return novelty

def score(row, prompt,past_sentences, novelty_weight=0.5, fluency_weight=0.5):
  if past_sentences is None:
    novelty_score = 1
  else:
    if row[prompt] != "" and isinstance(row[prompt], str):
        novelty_score = calculate_novelty(row[prompt], past_sentences=past_sentences)
    else:
        novelty_score = 0 
  if type(row[prompt])!=str or len(row[prompt]) < 3:
    fluency_score = 0
  else:
    fluency_score = row["ppl_"+prompt]
  print(f"{row[prompt]} has a novelty score {novelty_score} and fluency score {fluency_score} ")
  weighted_score = novelty_weight*novelty_score + fluency_weight*fluency_score
  return {"score": weighted_score}

def add_response_to_past_responses(latest_response, past_responses):
    latest_response_n_grams = generate_n_gram(latest_response, max_n_gram=3)
    if past_responses is None:
        past_responses = dict()
        past_responses[0] = latest_response_n_grams
    else:
        past_responses[len(past_responses)] = latest_response_n_grams
    return past_responses

def generate_next_response(prompt, past_sentences):
  dataset = load_from_disk('/Users/mariakosyuchenko/ChatBoth thesis/russian_chatbot/chatbot_main/russian_therapy/data/topKdataset')
  scores = dataset.map(lambda row: score(row, prompt, past_sentences), load_from_cache_file=False)
  result = scores.sort('score')
  return result[-1][prompt]

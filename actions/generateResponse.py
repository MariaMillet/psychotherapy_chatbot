def generate_N_grams(text,ngram=1):
  words=[word for word in text.split(" ")] 
  # print("Sentence after removing stopwords:",words)
  temp=zip(*[words[i:] for i in range(0,ngram)])
  ans=[' '.join(ngram) for ngram in temp]
  return ans    

def generate_n_gram(sentence, max_n_gram):
  n_gram_list = [generate_N_grams(sentence, n_gram) for n_gram in range(1, max_n_gram+1)]
  return n_gram_list

def novelty_score(sent_1_n_grams, sent_2_n_grams):
#   print(sent_1_n_grams, sent_2_n_grams)
  common_n_grams = [set(sent_1_n_grams[i]).intersection(set(sent_2_n_grams[i])) for i in range(len(sent_1_n_grams))]
  try:
    score = [len(common_n_grams[i])/min(len(sent_1_n_grams[i]), len(sent_2_n_grams[i])) for i in range(len(common_n_grams))]
    score =sum(score)/len(score)
  except ZeroDivisionError:
    score = 0

  return score

def calculate_novelty(sentence,past_sentences):
  new_sentence_n_grams = generate_n_gram(sentence, max_n_gram=3)
#   print(len(past_sentences))
  novelty = sum([novelty_score(new_sentence_n_grams, past_sentences[str(i)]) for i in range(len(past_sentences))])/len(past_sentences)
  return novelty

def score(row, prompt,past_sentences, novelty_weight=0.5, fluency_weight=0.5):
  if past_sentences is None:
    novelty_score = 0
  else:
    if row[prompt] != "" and isinstance(row[prompt], str):
        novelty_score = calculate_novelty(row[prompt], past_sentences=past_sentences)
    else:
        novelty_score = 0 
  if type(row[prompt])!=str or len(row[prompt]) < 3:
    fluency_score = 0
  else:
    fluency_score = row["ppl_"+prompt]
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

# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
from asyncio import protocols
import email
from typing import Any, Text, Dict, List
from datetime import datetime, timedelta

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk.events import FollowupAction
from rasa_sdk.events import UserUttered
from rasa_sdk.events import ActionExecuted
from rasa_sdk.events import SessionStarted
from rasa_sdk.events import EventType


import os
import pandas as pd 
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from transformers import BertForSequenceClassification, AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers import (
    glue_tasks_num_labels,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    GlueDataset,
    GlueDataTrainingArguments,
    TrainingArguments,
)
from sentence_transformers import SentenceTransformer, util

import nltk
from nltk.util import ngrams

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
import gc

import datasets
from datasets import load_dataset
from datasets import DatasetDict
from datasets.load import load_from_disk

import numpy as np
from numpy.lib.function_base import average

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

class EmotionClassifier:
    def __init__(self, model='mmillet/distilrubert-tiny-2ndfinetune-epru'):
        self.model_id = model
        self.classifier = pipeline('text-classification', model=self.model_id)
    
    def int2label(self, class_number):
        dict = {}
        labels = ['anger', 'fear', 'joy', 'sadness']
        for i in range(len(labels)):
            dict[i] = labels[i]
        return dict[class_number]

    def predict_emotion(self, text):
        preds = self.classifier(text, return_all_scores=True)
        # print(preds)
        # print(preds[0]['score'])
        prediction = max(preds, key=lambda x: x['score'])
        prediction_label = self.int2label(int(prediction['label'][-1]))
        return prediction_label

def calculate_similarity(row, querry_embedding):
    simil = util.cos_sim(row['embeddings'][0][0], querry_embedding)
    return {'similarity': simil}


def generate_response_dataset(k=5, querry="Чувствую себя очень плохо. Мне очень грустно."):
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
    model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
    pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    querry_embedding = pipe(querry)
    dataSet = load_from_disk('/Users/mariakosyuchenko/ChatBoth thesis/russian_chatbot/data/respGenDataset')
    ds= dataSet.map(lambda x: calculate_similarity(x, querry_embedding[0][0]))
    sorted_dataset = ds.sort('similarity')
    top_10_similar = sorted_dataset['train'].select([0,1,2,3,4,5,6,7,8,9,10])
    return top_10_similar

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
  common_n_grams = [set(sent_1_n_grams[i]).intersection(set(sent_2_n_grams[i])) for i in range(len(sent_1_n_grams))]
  score = [len(common_n_grams[i])/min(len(sent_1_n_grams[i]), len(sent_2_n_grams[i])) for i in range(len(common_n_grams))]
  return sum(score)/len(score)

def calculate_novelty(sentence,past_sentences):
  new_sentence_n_grams = generate_n_gram(sentence, max_n_gram=3)
  novelty = sum([novelty_score(new_sentence_n_grams, past_sentences[i]) for i in past_sentences])/len(past_sentences)
  return novelty

def score(row, prompt,past_sentences, novelty_weight=0.5, fluency_weight=0.5):
  novelty_score = calculate_novelty(row[prompt], past_sentences=past_sentences)
  fluency_score = row["ppl_"+prompt]
  weighted_score = novelty_weight*novelty_score + fluency_weight*fluency_score
  return {"score": weighted_score}

def generate_next_response(prompt, past_sentences):
  dataset = load_from_disk('/Users/mariakosyuchenko/ChatBoth thesis/russian_chatbot/data/topKdataset')
  scores = dataset.map(lambda row: score(row, prompt, past_sentences))
  result = scores.sort('score')
  return result[-1][prompt]


# class ResponsesSlot(Slot):
#     def __init__(self):
#         self.responses = {}
#         self.n = 0



class AskForSlotActionFeeling(Action):
    def name(self) -> Text:
        return "action_ask_current_feeling"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        dispatcher.utter_message(text="Please tell me how are you feeling now")
        
        dict = {}
        dict[1] = ['hello']
        return []

class AskForSlotActionEmotioPrediction(Action):
    def name(self) -> Text:
        return "action_ask_emotion_prediction"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        example = EmotionClassifier()
        current_feeling = tracker.get_slot("current_feeling")
        pred = example.predict_emotion(current_feeling)
        dispatcher.utter_message(f"I see your emotion as '{pred}'")
        dataset = generate_response_dataset()
        print(type(dataset))
        sad_d = dataset.save_to_disk('/Users/mariakosyuchenko/ChatBoth thesis/russian_chatbot/data/topKdataset')

        # if tracker.get_slot("emotion_confirmation")=='No':
        #     print('success')
        # else:
        #     print("fail")
        return [SlotSet("emotion_prediction",pred),FollowupAction("action_emotion_confirmation")]

class AskForSlotActionEmotionConfirmation(Action):
    def name(self) -> Text:
        return "action_emotion_confirmation"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        # print(tracker.get_slot("emotion_confirmation"))
        # test = tracker.get_slot("response")
        # print(test)
        emotion_prediction = tracker.get_slot("emotion_prediction")
        # intent = tracker.get_intent_of_latest_message()
        # print(intent)
        # if intent == "affirm":
        #     em_conf = "yes"
        # else:
        #     em_conf = "no"
        # print(em_conf)
        # buttons = [{"title": "Yes, this is accurate", "payload": '/affirm{"emotion_confirmation":"Yes"}'}, {"title": "No", "payload": '/deny{"emotion_confirmation":"No"}'}]
        buttons = [{"title": "Yes, this is accurate", "payload": '/correct_prediction{"emotion":"emotion_prediction"}'}, {"title": "No", "payload":'/not_correct_prediction'}]
        dispatcher.utter_message(text=f"Please confirm if your emotion is {emotion_prediction}, Did I understand your feeling correct? ", buttons=buttons)
        # print(tracker.get_slot('emotion_confirmation'))
        return []
        # return [SlotSet("emotion_confirmation",pred)]

class AskForSlotActionEmotionConfirmation(Action):
    def name(self) -> Text:
        return "action_manual_emotion_selection"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        buttons = [{"title": "joy", "payload": '/is_recent{"emotion":"joy"}'}, {"title": "anger", "payload":'/is_recent{"emotion":"anger"}'}]
        dispatcher.utter_button_message(f"I am sorry, please select manually", buttons)
        # print(tracker.get_slot('emotion_confirmation'))
        return []

class AskIfEventWasRecent(Action):
    def name(self) -> Text:
        return "action_is_recent"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        # buttons = [{"title": "Yes", "payload": '/is_recent{"emotion":"joy"}'}, {"title": "anger", "payload":'/is_recent{"emotion":"joy"}'}]
        # dispatcher.utter_button_message(f"I am sorry, please select manually", buttons)
        dispatcher.utter_message(f"enough for today")
        em = tracker.get_slot("emotion")
        buttons = [{"title": "yes", "payload": '/is_protocol_6_distressing'}, {"title": "no", "payload":'/is_protocol_11_distressing'}]
        dispatcher.utter_button_message(f"Was this a recent event {em}", buttons)
        first_message = dict()
        first_message[1] = "Was this a recent event"
        # print(tracker.get_slot('emotion_confirmation'))
        return [SlotSet('pastResponses',first_message)]

class Action6distressing(Action):
    def name(self) -> Text:
        return "action_is_protocol_6_distressing"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        prot = [13, 17]
        print(tracker.get_slot('pastResponses'))
        past_responses = tracker.get_slot('pastResponses')
        test = generate_next_response(prompt="how_recent_is_the_event?", past_sentences=past_responses)
        print(test)
        buttons = [{"title": "yes", "payload": '/ok_to_ask_more{"protocols_1":["13","25"]}'}, {"title": "no", "payload":'/ok_to_ask_more{"protocols_1":[6]}'}]
        dispatcher.utter_button_message(f"Did you find protocol 6 distressing?", buttons)
        return []

class AskMoreQuestions(Action):
    def name(self) -> Text:
        return "action_ok_to_ask_more"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        # buttons = [{"title": "Yes", "payload": '/is_recent{"emotion":"joy"}'}, {"title": "anger", "payload":'/is_recent{"emotion":"joy"}'}]
        # dispatcher.utter_button_message(f"I am sorry, please select manually", buttons)
        protocols_so_far = tracker.get_slot("protocols_1")
        em = tracker.get_slot("emotion")
        dispatcher.utter_message(f"enough for today{protocols_so_far} and {type(protocols_so_far)} so far and {em}")
        buttons = [{"title": "yes", "payload": '/strong_emotions'}, {"title": "no", "payload":'/strong_emiotions'}]
        dispatcher.utter_button_message(f"is it ok to ask more questions", buttons)
        # buttons = [{"title": "yes", "payload": '/ok_to_ask_more'}, {"title": "no", "payload":'/ok_to_ask_more'}]
        # dispatcher.utter_button_message(f"Was this a recent event", buttons)
        # print(tracker.get_slot('emotion_confirmation'))
        return []

class AdditionalQuestionsStrongEmotion(Action):
    def name(self) -> Text:
        return "action_strong_emotions"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        # buttons = [{"title": "Yes", "payload": '/is_recent{"emotion":"joy"}'}, {"title": "anger", "payload":'/is_recent{"emotion":"joy"}'}]
        # dispatcher.utter_button_message(f"I am sorry, please select manually", buttons)
        protocols_so_far = tracker.get_slot("protocols_1")
        print(f"protocols {protocols_so_far} and {type(protocols_so_far)}")
        em = tracker.get_slot("emotion")
        message = "Have you strongly expressed following emotions?"
        # new_protocols = protocols_so_far.copy()
        # new_protocols.append(25)
        # print(type(new_protocols))
        # print(new_protocols)
        new_protocols = 7

        buttons = [{"title": "yes", "payload": '/recommend_protocols{"protocols_2":[13,14]}'}, {"title": "no", "payload":'/ok_to_ask_more'}]
        dispatcher.utter_button_message(message, buttons)
        # buttons = [{"title": "yes", "payload": '/ok_to_ask_more'}, {"title": "no", "payload":'/ok_to_ask_more'}]
        # dispatcher.utter_button_message(f"Was this a recent event", buttons)
        # print(tracker.get_slot('emotion_confirmation'))
        return []




class ActionRecommendProtocols(Action):
    def name(self) -> Text:
        return "action_recommend_protocols"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        # buttons = [{"title": "Yes", "payload": '/is_recent{"emotion":"joy"}'}, {"title": "anger", "payload":'/is_recent{"emotion":"joy"}'}]
        # dispatcher.utter_button_message(f"I am sorry, please select manually", buttons)

        protocols = tracker.get_slot("protocols_1") + tracker.get_slot("protocols_2")
        dispatcher.utter_message(f"I would like to recommend the following protocols {protocols}")
        # buttons = [{"title": "yes", "payload": '/ok_to_ask_more'}, {"title": "no", "payload":'/ok_to_ask_more'}]
        # dispatcher.utter_button_message(f"Was this a recent event", buttons)
        # print(tracker.get_slot('emotion_confirmation'))
        return []
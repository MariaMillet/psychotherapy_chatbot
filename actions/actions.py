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

from actions.generateResponse import *

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


def generate_next_response(prompt, past_sentences):
  dataset = load_from_disk('/Users/mariakosyuchenko/ChatBoth thesis/russian_chatbot/data/topKdataset')
  scores = dataset.map(lambda row: score(row, prompt, past_sentences))
  result = scores.sort('score')
  return result[-1][prompt]


class AskForSlotActionFeeling(Action):
    def name(self) -> Text:
        return "action_ask_current_feeling"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        dispatcher.utter_message(text="Please tell me how are you feeling now")
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
        buttons = [{"title": "Yes, this is accurate", "payload": '/correct_prediction{"emotion":"emotion_prediction"}'}, {"title": "No", "payload":'/not_correct_prediction'}]
        dispatcher.utter_message(text=f"Please confirm if your emotion is {emotion_prediction}, Did I understand your feeling correct? ", buttons=buttons)
        return []

class AskForSlotActionEmotionConfirmation(Action):
    def name(self) -> Text:
        return "action_manual_emotion_selection"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        buttons = [{"title": "joy", "payload": '/is_recent{"emotion":"joy"}'}, {"title": "anger", "payload":'/is_recent{"emotion":"anger"}'}]
        dispatcher.utter_button_message(f"I am sorry, please select manually", buttons)
        dataset = generate_response_dataset()
        sad_d = dataset.save_to_disk('/Users/mariakosyuchenko/ChatBoth thesis/russian_chatbot/data/topKdataset')
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
        buttons = [{"title": "yes", "payload": '/is_protocol_6_distressing'}, {"title": "no", "payload":'/is_protocol_11_distressing'}]
        text = generate_next_response(prompt="how_recent_is_the_event?", past_sentences=tracker.get_slot('pastResponses'))
        print(text)
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        dispatcher.utter_message(text=text, buttons=buttons)
        return [SlotSet('pastResponses',updated_responses)]

class Action6distressing(Action):
    def name(self) -> Text:
        return "action_is_protocol_6_distressing"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        text = generate_next_response(prompt="method_6", past_sentences=tracker.get_slot('pastResponses'))
        print(text)
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        buttons = [{"title": "yes", "payload": '/ok_to_ask_more{"protocols_1":["13","25"]}'}, {"title": "no", "payload":'/ok_to_ask_more{"protocols_1":[6]}'}]
        dispatcher.utter_message(text=text, buttons=buttons)
        return [SlotSet('pastResponses',updated_responses)]

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
        text = generate_next_response(prompt="additional_questions_permission", past_sentences=tracker.get_slot('pastResponses'))
        print(text)
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        buttons = [{"title": "yes", "payload": '/strong_emotions'}, {"title": "no", "payload":'/strong_emiotions'}]
        dispatcher.utter_message(text=text, buttons=buttons)
        past_responses = tracker.get_slot('pastResponses')
        print(past_responses)
        # buttons = [{"title": "yes", "payload": '/ok_to_ask_more'}, {"title": "no", "payload":'/ok_to_ask_more'}]
        # dispatcher.utter_button_message(f"Was this a recent event", buttons)
        # print(tracker.get_slot('emotion_confirmation'))
        return [SlotSet('pastResponses',updated_responses)]

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
        text = generate_next_response(prompt="ppl_feeling_envy_mistrust_trigger", past_sentences=tracker.get_slot('pastResponses'))
        print(text)
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        # new_protocols = protocols_so_far.copy()
        # new_protocols.append(25)
        # print(type(new_protocols))
        # print(new_protocols)
        new_protocols = 7

        buttons = [{"title": "yes", "payload": '/recommend_protocols{"protocols_2":[13,14]}'}, {"title": "no", "payload":'/ok_to_ask_more'}]
        dispatcher.utter_message(text=text, buttons=buttons)
        # buttons = [{"title": "yes", "payload": '/ok_to_ask_more'}, {"title": "no", "payload":'/ok_to_ask_more'}]
        # dispatcher.utter_button_message(f"Was this a recent event", buttons)
        # print(tracker.get_slot('emotion_confirmation'))
        return [SlotSet('pastResponses',updated_responses)]




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
from asyncio import protocols
import email
from typing import Any, Text, Dict, List
from datetime import datetime, timedelta
from matplotlib.font_manager import json_dump

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk.events import FollowupAction
from rasa_sdk.events import UserUttered
from rasa_sdk.events import ActionExecuted
from rasa_sdk.events import SessionStarted
from rasa_sdk.events import EventType
from rasa_sdk.events import AllSlotsReset

from actions.generateResponse import *

import os
# import pandas as pd 
# import matplotlib.pyplot as plt

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

from numpy import random

import nltk
from nltk.util import ngrams

import json

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
    dataSet = load_from_disk('/Users/mariakosyuchenko/ChatBoth thesis/russian_chatbot/chatbot_main/russian_therapy/data/respGenDataset')
    ds= dataSet.map(lambda x: calculate_similarity(x, querry_embedding[0][0]))
    sorted_dataset = ds.sort('similarity')
    top_10_similar = sorted_dataset['train'].select([0,1,2,3,4,5,6,7,8,9,10])
    return top_10_similar


def generate_next_response(prompt, past_sentences):
  dataset = load_from_disk('/Users/mariakosyuchenko/ChatBoth thesis/russian_chatbot/chatbot_main/russian_therapy/data/topKdataset')
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
        prompts = ['Have you strongly felt or expressed any of the following emotions towards someone?', 'Do you believe you should be the savior of someone else?', 'Do you see yourself as the victim, blaming someone else for how negative you feel?',
            'Do you feel that you are trying to control someone?', 'Are you always blaming and accusing yourself for when something goes wrong?', 'Is it possible that in previous conversations you may not have always considerdother viewpoints presented?',
            'Are you undergoing a personal crisis (experience difficulties with loved ones e.g. falling out with friends)?']
        return [SlotSet("additionalQuestionsPrompts", prompts)]


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
        dataset = generate_response_dataset(k=5, querry=tracker.get_slot('current_feeling'))
        sad_d = dataset.save_to_disk('/Users/mariakosyuchenko/ChatBoth thesis/russian_chatbot/chatbot_main/russian_therapy/data/topKdataset')
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
        text = generate_next_response(prompt="Was the event recent?", past_sentences=tracker.get_slot('pastResponses'))
        # print(text)
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        dispatcher.utter_message(text=text, buttons=buttons)
        return [SlotSet('pastResponses',updated_responses)]

class Action6distressing(Action):
    def name(self) -> Text:
        return "action_is_protocol_6_distressing"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        print(tracker.latest_message['text'])
        # for event in (list(reversed(tracker.events)))[:5]: # latest 5 messages
        #     if event.get("event") == "user": # check if the sent by user or bot
        #         print(event.get("text"))
        text = generate_next_response(prompt="Did you find protocol 6 distressing?", past_sentences=tracker.get_slot('pastResponses'))
        # print(text)
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        dictionary ={"protocols_1":"['13,'25']"}
        json_object = json.dumps(dictionary, indent = 4) 
        # print(json_object)
        buttons = [{"title": "yes, this event is recent", "payload": '/ok_to_ask_more{"protocols_1":[13,25]}'}, {"title": "no", "payload":'/ok_to_ask_more{"protocols_1":[6]}'}]
        dispatcher.utter_message(text=text, buttons=buttons)
        return [SlotSet('pastResponses',updated_responses)]

class AskMoreQuestions(Action):
    def name(self) -> Text:
        return "action_ok_to_ask_more"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        print(tracker.latest_message['text'])
        for event in (list(reversed(tracker.events)))[:5]: # latest 5 messages
            if event.get("event") == "user": # check if the sent by user or bot
                print(event.get("text"))
        protocols_so_far = tracker.get_slot("protocols_1")
        em = tracker.get_slot("emotion")
        text = generate_next_response(prompt="Is it ok to ask additional questions?", past_sentences=tracker.get_slot('pastResponses'))
        # print(text)
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        buttons = [{"title": "yes", "payload": '/additional_questions'}, {"title": "no", "payload":'/additional_questions'}]
        dispatcher.utter_message(text=text, buttons=buttons)
        past_responses = tracker.get_slot('pastResponses')
        # print(past_responses)
        # buttons = [{"title": "yes", "payload": '/ok_to_ask_more'}, {"title": "no", "payload":'/ok_to_ask_more'}]
        # dispatcher.utter_button_message(f"Was this a recent event", buttons)
        # print(tracker.get_slot('emotion_confirmation'))
        return [SlotSet('pastResponses',updated_responses)]

class AdditionalQuestions(Action):
    def name(self) -> Text:
        return "action_additional_questions"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        available_prompts = tracker.get_slot("additionalQuestionsPrompts")
        print(available_prompts)
        prompt = random.choice(available_prompts)
        text = generate_next_response(prompt=prompt, past_sentences=tracker.get_slot('pastResponses'))
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        if len(available_prompts) == 1:
            buttons = [{"title": "yes", "payload":'/recommend_protocols{"positive_to_any_base_questions": "True"}'}, {"title": "no", "payload":'/recommend_protocols{"positive_to_any_base_questions": "False"}'}]
        else:
            buttons = [{"title": "yes", "payload":'/recommend_protocols{"positive_to_any_base_questions": "True"}'}, {"title": "no", "payload":'/additional_questions'}]   
            available_prompts.remove(prompt)
        dispatcher.utter_message(text=text, buttons=buttons)

        return [SlotSet("additionalQuestionsPrompts", available_prompts), SlotSet('pastResponses',updated_responses), SlotSet('lastPrompt', prompt)] 

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
        text = generate_next_response(prompt="Have you strongly felt or expressed any of the following emotions towards someone?", past_sentences=tracker.get_slot('pastResponses'))
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
        positive_to_any_base_questions = tracker.get_slot("positive_to_any_base_questions")

        prompt_mapping = {'Have you strongly felt or expressed any of the following emotions towards someone?':[13,14], 'Do you believe you should be the savior of someone else?':[8,15,16,19], 'Do you see yourself as the victim, blaming someone else for how negative you feel?':[8,15,16,19],
            'Do you feel that you are trying to control someone?':[8,15,16,19], 'Are you always blaming and accusing yourself for when something goes wrong?':[8,15,16,19], 'Is it possible that in previous conversations you may not have always considerdother viewpoints presented?':[13,19],
            'Are you undergoing a personal crisis (experience difficulties with loved ones e.g. falling out with friends)?':[13,19]}

        protocols = tracker.get_slot("protocols_1") 
        
        if positive_to_any_base_questions == "True":
            protocols_additional_questions = prompt_mapping[tracker.get_slot('lastPrompt')]
            dispatcher.utter_message(f"I would like to recommend the following protocols {protocols}")
            protocols = protocols_additional_questions + protocols

        [{"title": "yes", "payload": '/is_protocol_6_distressing'}, {"title": "no", "payload":'/is_protocol_11_distressing'}]
        buttons = []
        for protocol in protocols:
            button = dict()
            button["title"] = protocol
            number = dict()
            number["number"] = protocol
            d = json.dumps(number)
            button["payload"] = f'/invite_to_protocol{d}'
            buttons.append(button)
        print(buttons)
        dispatcher.utter_message(text = f"Спасибо, что поделились со мной. Я бы хотел предложить вам несколько методик", buttons=buttons)
        # buttons = [{"title": "yes", "payload": '/ok_to_ask_more'}, {"title": "no", "payload":'/ok_to_ask_more'}]
        # dispatcher.utter_button_message(f"Was this a recent event", buttons)
        # print(tracker.get_slot('emotion_confirmation'))
        return [SlotSet('relevant_protocols', protocols)]

class ActionInviteToProtocols(Action):
    def name(self) -> Text:
        return "action_invite_to_protocol"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        protocol_number = tracker.get_slot("number")
        relevant_protocols = tracker.get_slot("relevant_protocols")
        if len(relevant_protocols) == 1:
            relevant_protocols = []
        else:
            relevant_protocols.remove(protocol_number)
        buttons = [{"title": "continue", "payload": '/ask_for_feedback'}]
        # dispatcher.utter_message(text=f"Look at that baby tiger {protocol_number}!", image = "https://i.imgur.com/nGF1K8f.jpg", buttons=buttons)
        dispatcher.utter_message(text=f"Please press continue when you are ready", buttons=buttons)
        return [SlotSet('relevant_protocols', relevant_protocols)]

class ActionAskForFeedback(Action):
    def name(self) -> Text:
        return "action_ask_for_feedback"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        protocol_number = tracker.get_slot("number")
        buttons = [{"title": "Better", "payload": '/respond_to_feedback{"response_type":"positive"}'}, {"title": "Same", "payload": '/respond_to_feedback{"response_type":"encouraging"}'}, {"title": "Worse", "payload": '/respond_to_feedback{"response_type":"encouraging"}'} ]
        dispatcher.utter_message(text=f"Thank you for attempting a protocol, how are you feeling now?", buttons=buttons)
        return [SlotSet("current_feeling", None), SlotSet("emotion_prediction", None)]

class ActionRespondToFeedback(Action):
    def name(self) -> Text:
        return "action_respond_to_feedback"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        response_type = tracker.get_slot("response_type")
        if response_type == "positive":
            text = "I am delighted you are feeling better. Would you like to try any other protocols?"
        if response_type == "Same" or response_type == "Worse":
            text = "I am sorry it did not help. Would you like to try any more protocols?"
        
        buttons = [{"title": "No", "payload": '/end_session'}]

        yes_button = dict()
        yes_button["title"] = "Yes"
        
        relevant_protocols = tracker.get_slot("relevant_protocols")
        if len(relevant_protocols) >= 1:
            yes_button["payload"] = '/protocol_recommendation_follow_up'
        else:
            yes_button["payload"] = '/greet'

        buttons.append(yes_button) 

        # buttons = [{"title": "Yes", "payload": '/generate_response{"response_type":"positive"}'}, {"title": "No", "payload": '/generate_response{"response_type":"encouraging"}'}, {"title": "Worse", "payload": '/generate_response{"response_type":"encouraging"}'} ]
        dispatcher.utter_message(text=text, buttons=buttons)
        return []
    
class ActionRecommendProtocols(Action):
    def name(self) -> Text:
        return "action_protocol_recommendation_follow_up"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:


        protocols = tracker.get_slot("relevant_protocols") 
        

        buttons = []
        for protocol in protocols:
            button = dict()
            button["title"] = protocol
            number = dict()
            number["number"] = protocol
            d = json.dumps(number)
            button["payload"] = f'/invite_to_protocol{d}'
            buttons.append(button)
        print(buttons)
        dispatcher.utter_message(text = f"Please choose one of the following protocols", buttons=buttons)
        # buttons = [{"title": "yes", "payload": '/ok_to_ask_more'}, {"title": "no", "payload":'/ok_to_ask_more'}]
        # dispatcher.utter_button_message(f"Was this a recent event", buttons)
        # print(tracker.get_slot('emotion_confirmation'))
        return []


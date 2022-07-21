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
from actions.classifyEmotion import *
# from actions.generateResponsePool import *

import os

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


def calculate_similarity(row, querry_embedding):
    simil = util.cos_sim(row['embeddings'][0][0], querry_embedding)
    return {'similarity': simil}


def generate_response_dataset(k=5, querry="Чувствую себя очень плохо. Мне очень грустно.", emotion='anger'):
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
    model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
    pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    querry_embedding = pipe(querry)
    dataSet = load_from_disk('/Users/mariakosyuchenko/ChatBoth thesis/russian_chatbot/chatbot_main/russian_therapy/data/respGenDataset')
    subsetDataSet = dataSet.filter(lambda row: row['emotion']==emotion)
    ds= subsetDataSet.map(lambda x: calculate_similarity(x, querry_embedding[0][0]), load_from_cache_file=False)
    sorted_dataset = ds.sort('similarity', 
    load_from_cache_file=False, reverse=True)
    top_10_similar = sorted_dataset['train'].select([0,1,2,3,4,5,6,7,8,9,10])
    return top_10_similar


def generate_next_response(prompt, past_sentences):
  dataset = load_from_disk('/Users/mariakosyuchenko/ChatBoth thesis/russian_chatbot/chatbot_main/russian_therapy/data/topKdataset')
  scores = dataset.map(lambda row: score(row, prompt, past_sentences), load_from_cache_file=False)
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

        emotion_prediction = tracker.get_slot("emotion_prediction")
        buttons = [{"title": "Да мне близко чувство", "payload": '/correct_prediction{"emotion":"emotion_prediction"}'}, {"title": "Нет, вы не угадали мое настроение", "payload":'/not_correct_prediction'}]
        dispatcher.utter_message(text=f"Мне кажется вы пребываете в {emotion_prediction} расположении духа, я правильно вас понял?", buttons=buttons)
        return []

class AskForSlotActionEmotionConfirmation(Action):
    def name(self) -> Text:
        return "action_manual_emotion_selection"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        buttons = [{"title": "joy", "payload": '/is_recent{"emotion":"joy"}'}, {"title": "anger", "payload":'/is_recent{"emotion":"anger"}'}, {"title": "fear", "payload":'/is_recent{"emotion":"fear"}'}, {"title": "sadness", "payload":'/is_recent{"emotion":"sadness"}'}]
        dispatcher.utter_button_message(f"Извиняюсь, Маше ещё надо поработать над маштнным обучением :-) Выберите, пожалуйста, подходящую эмоцию 'вручную'", buttons)

        return []

class AskIfEventWasRecent(Action):
    def name(self) -> Text:
        return "action_is_recent"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        # Create a subset of EPRU dataset as measured by the similarity of user utterance to emotion utterances in the dataset for an emotion specified
        dataset = generate_response_dataset(k=5, querry=tracker.get_slot('current_feeling'), emotion=tracker.get_slot('emotion'))
        dataset.save_to_disk('/Users/mariakosyuchenko/ChatBoth thesis/russian_chatbot/chatbot_main/russian_therapy/data/topKdataset')

        buttons = [{"title": "Да, эта ситуация произошла недавно", "payload": '/is_protocol_6_distressing'}, {"title": "Нет, это случилочь достаточно давно", "payload":'/is_protocol_11_distressing'}]
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

        text = generate_next_response(prompt="Did you find protocol 6 distressing?", past_sentences=tracker.get_slot('pastResponses'))
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        buttons = [{"title": "Согласен, этот метод меня немного встревожил", "payload": '/ok_to_ask_more{"protocols_1":[13,7]}'}, {"title": "Нет, это упражнение не вызвало у меня волнения", "payload":'/ok_to_ask_more{"protocols_1":[6]}'}]
        dispatcher.utter_message(text=text, buttons=buttons)
        return [SlotSet('pastResponses',updated_responses)]

class AskMoreQuestions(Action):
    def name(self) -> Text:
        return "action_ok_to_ask_more"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:


        text = generate_next_response(prompt="Is it ok to ask additional questions?", past_sentences=tracker.get_slot('pastResponses'))
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        buttons = [{"title": "Я не против", "payload": '/additional_questions'}, {"title": "Нет, на сегодня достаточно вопросов", "payload":'/additional_questions'}]
        dispatcher.utter_message(text=text, buttons=buttons)

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
            buttons = [{"title": "Больше да, чем нет", "payload":'/recommend_protocols{"positive_to_any_base_questions": "True"}'}, {"title": "Думаю нет", "payload":'/recommend_protocols{"positive_to_any_base_questions": "False"}'}]
        else:
            buttons = [{"title": "Больше да, чем нет", "payload":'/recommend_protocols{"positive_to_any_base_questions": "True"}'}, {"title": "Нет", "payload":'/additional_questions'}]   
            available_prompts.remove(prompt)
        dispatcher.utter_message(text=text, buttons=buttons)

        return [SlotSet("additionalQuestionsPrompts", available_prompts), SlotSet('pastResponses',updated_responses), SlotSet('lastPrompt', prompt)] 


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
            protocols = protocols_additional_questions + protocols

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
        dispatcher.utter_message(text = f"Спасибо, что уделили мне ваше ценное время. Исходя из нашей беседы, я подготовил несколько методик на ваше усмотрение.", buttons=buttons)
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
        buttons = [{"title": "Продолжить", "payload": '/ask_for_feedback'}]
        # dispatcher.utter_message(text=f"Look at that baby tiger {protocol_number}!", image = "https://i.imgur.com/nGF1K8f.jpg", buttons=buttons)
        dispatcher.utter_message(text=f"Пожалуйта, нажмите 'Продолжить', когда будете готовы", buttons=buttons)
        return [SlotSet('relevant_protocols', relevant_protocols)]

class ActionAskForFeedback(Action):
    def name(self) -> Text:
        return "action_ask_for_feedback"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        protocol_number = tracker.get_slot("number")
        buttons = [{"title": "Better", "payload": '/respond_to_feedback{"response_type":"positive"}'}, {"title": "Same", "payload": '/respond_to_feedback{"response_type":"encouraging_same"}'}, {"title": "Worse", "payload": '/respond_to_feedback{"response_type":"encouraging_worse"}'} ]
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
        if response_type == "encouraging_same":
            text = "I am sorry it did not help. Would you like to try any more protocols?"
        if response_type == "encouraging_worse":
            text = "I am sorry it did make you feel better. Would you like to try any more protocols?"
        
        buttons = [{"title": "No", "payload": '/goodbye'}]

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
        dispatcher.utter_message(text = f"Представляю на выбор несколько методик", buttons=buttons)

        return []


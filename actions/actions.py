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
    dataSet = load_from_disk('./data/respGenDataset')
    subsetDataSet = dataSet.filter(lambda row: row['emotion']==emotion)
    ds= subsetDataSet.map(lambda x: calculate_similarity(x, querry_embedding[0][0]), load_from_cache_file=False)
    sorted_dataset = ds.sort('similarity', 
    load_from_cache_file=False, reverse=True)
    top_10_similar = sorted_dataset['train'].select([0,1,2,3,4,5,6,7,8,9,10])
    return top_10_similar

def generate_synthetic_dataset(k=10, emotion='anger'):
    dataSet = load_from_disk('./data/respSyntheticDataset')
    subsetDataSet = dataSet.filter(lambda row: row['emotion']==emotion)
    if emotion == "joy":
        max_row = len(subsetDataSet["Would you like to attempt protocols for joy?"])
    else:
        max_row = len(subsetDataSet["Did you find protocol 6 distressing?"])
    random_selection = subsetDataSet.select(random.choice(max_row, 10, replace=False))
    print(random_selection)
    return random_selection



def generate_next_response(prompt, past_sentences, type, novelty_weight=0.5, fluency_weight=0.25, empathy_weight=0.25, empathy_mode="medium"):
  if type=="synthetic":
    dataset = load_from_disk('./data/randomSyntheticDataset')
    print(dataset[0]["Would you like to attempt protocols for joy?"])
  else:
    dataset = load_from_disk('./data/topKdataset')
  scores = dataset.map(lambda row: score(row, prompt, past_sentences, novelty_weight, fluency_weight, empathy_weight, empathy_mode=empathy_mode), load_from_cache_file=False)
  result = scores.sort('score')
  return result[-1][prompt]





class AskForSlotActionFeeling(Action):
    def name(self) -> Text:
        return "action_ask_personality"


    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        # dispatcher.utter_message(text=f"Рад вас видеть, как поживаете? \u270F\uFE0F")
        # dispatcher.utter_message(text=f"Please select who you want to speak with \u270F\uFE0F")
        buttons = [{"title": "Кирилл - выражает свои мысли ясно, логично и предельно кратко \U0001F913", "payload": '/personality{"personality":"Кирилл", "response_type":"human", "empathy_mode":"medium"}'}, {"title": "Наташа - сама доброта, общается максимаально добро и ласково \U0001F60D ", "payload": '/personality{"personality":"Наташа", "response_type":"human", "empathy_mode":"high"}'}, {"title": "Компьюша - наиболее непредскауемый робот и наболее креативный в генерации ответов. Не судите его  строго, если иногда у него не всё получается \U0001F910 ", "payload": '/personality{"personality":"Компьюша", "response_type":"synthetic", "empathy_mode":"high"}'}]
        dispatcher.utter_message(text=f"Пожалуста, выберите с кем из наших умных и человечных \U0001F607 психологов вы бы хотели попобщаться", buttons=buttons)

        
        prompts = ['Have you strongly felt or expressed any of the following emotions towards someone?', 'Do you believe you should be the savior of someone else?', 'Do you see yourself as the victim, blaming someone else for how negative you feel?',
            'Do you feel that you are trying to control someone?', 'Are you always blaming and accusing yourself for when something goes wrong?', 'Is it possible that in previous conversations you may not have always considerdother viewpoints presented?',
            'Are you undergoing a personal crisis (experience difficulties with loved ones e.g. falling out with friends)?']
        return []

class AskForSlotActionFeeling(Action):
    def name(self) -> Text:
        return "action_ask_name"


    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        # dispatcher.utter_message(text=f"Рад вас видеть, как поживаете? \u270F\uFE0F")
        psychologist = tracker.get_slot("personality")
        type = tracker.get_slot("response_type")
        dispatcher.utter_message(text=f"Добрый день, я виртуальный психолог, меня зовут {psychologist}. Как я могу к вам обращаться?")
        print(tracker.get_slot("empathy_mode"))
        # buttons = [{"title": "Kirill", "payload": '/joy_enquire_for_protocol{"emotion":"joy"}'}, {"title": "Natasha", "payload":'/not_correct_prediction'}]
        # prompts = ['Have you strongly felt or expressed any of the following emotions towards someone?', 'Do you believe you should be the savior of someone else?', 'Do you see yourself as the victim, blaming someone else for how negative you feel?',
            # 'Do you feel that you are trying to control someone?', 'Are you always blaming and accusing yourself for when something goes wrong?', 'Is it possible that in previous conversations you may not have always considerdother viewpoints presented?',
            # 'Are you undergoing a personal crisis (experience difficulties with loved ones e.g. falling out with friends)?']
        # if len(tracker.get_slot(name).split())!=1:
        #     dispatcher.utter_message(text="sorry i am only trained to understand a signle word names")
        #     name = None
        # else:
        #     name = tracker.get_slot("name")
        return []

class AskForSlotActionFeeling(Action):
    def name(self) -> Text:
        return "action_ask_current_feeling"


    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        if len(tracker.get_slot("name").split())!=1:
            dispatcher.utter_message(text="sorry i am only trained to understand a signle word names")
        else:
            name = tracker.get_slot("name")
        dispatcher.utter_message(text=f"Очень приятно, {name}! Я практикую методику SAT, и сделаю всё, чтобы улучшить ваше душевное состояние. Для начала, опишите ,пожалуйста, как вы себя ощущаете? \u270F\uFE0F ")
        # dispatcher.utter_message(text=f"Please select who you want to speak with \u270F\uFE0F", buttons=buttons)
        # buttons = [{"title": "yes", "payload": '/joy_enquire_for_protocol{"emotion":"joy"}'}, {"title": "Нет, вы не угадали мое настроение", "payload":'/not_correct_prediction'}]
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

        return [SlotSet("emotion_prediction",pred),FollowupAction("action_emotion_confirmation")]

class AskForSlotActionEmotionConfirmation(Action):
    def name(self) -> Text:
        return "action_emotion_confirmation"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        emotion_prediction = tracker.get_slot("emotion_prediction")
        eng_to_ru = {'joy': 'радостном', 'anger': 'разозлённом', 'sadness': 'грустном', 'fear': 'встревоженном'}
        ru_emotion_pred = eng_to_ru[emotion_prediction]
        if emotion_prediction == "joy":
            buttons = [{"title": "Да, мне близко это чувство", "payload": '/joy_enquire_for_protocol{"emotion":"joy"}'}, {"title": "Нет, вы не угадали мое настроение", "payload":'/not_correct_prediction'}]
        else:
            em = dict()
            em["emotion"] = tracker.get_slot('emotion_prediction')
            d = json.dumps(em)
            # button["payload"] = f'/invite_to_protocol{d}'
            buttons = [{"title": "Да, мне близко это чувство", "payload": f'/is_event{d}'}, {"title": "Нет, вы не угадали мое настроение", "payload":'/not_correct_prediction'}]
        
        dispatcher.utter_message(text=f"Спасибо, что поделились своими чувствами. Мне кажется вы пребываете в {ru_emotion_pred} расположении духа, я правильно вас понял?", buttons=buttons, button_type="vertical")
        return []

class AskForSlotActionEmotionConfirmation(Action):
    def name(self) -> Text:
        return "action_manual_emotion_selection"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        emotions = dict()
        emotions['joy'] = [f"Радостно \U0001F600", "joy_enquire_for_protocol"]
        buttons = [ {"title": f"Радостно \U0001F600 ", "payload":'/joy_enquire_for_protocol{"emotion":"joy"}'}, {"title": f"Ярость \U0001F621", "payload":'/is_event{"emotion":"anger"}'}, {"title": f"Тревога \U0001F628", "payload":'/is_event{"emotion":"fear"}'}, {"title": f"Грусть \U0001F622", "payload":'/is_event{"emotion":"sadness"}'}]

        dispatcher.utter_button_message(f"Извиняюсь, Маше ещё надо поработать над машинным обучением :-) Выберите, пожалуйста, подходящую эмоцию 'вручную'", buttons)
# \U0001F600 
        return []

class AskIfEventTriggeredEmotion(Action):
    def name(self) -> Text:
        return "action_is_event"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        text = tracker.get_slot("emotion")
        dispatcher.utter_message(text=text)
        # Create a subset of EPRU dataset as measured by the similarity of user utterance to emotion utterances in the dataset for an emotion specified
        if tracker.get_slot('personality') == "Компьюша":
            synthetic_dataset = generate_synthetic_dataset(k=10,emotion=tracker.get_slot('emotion'))
            synthetic_dataset.save_to_disk('./data/randomSyntheticDataset')
        else:
            dataset = generate_response_dataset(k=5, querry=tracker.get_slot('current_feeling'), emotion=tracker.get_slot('emotion'))
            dataset.save_to_disk('./data/topKdataset')

        buttons = [{"title": "Да, мои переживания связанны с событием из жизни", "payload": '/is_recent'}, {"title": "Нет, просто всё накатило", "payload":'/ok_to_ask_more'}]
        empathy_mode = tracker.get_slot("empathy_mode")
        text = generate_next_response(prompt="Was this emotion triggered by a specific event?", past_sentences=tracker.get_slot('pastResponses'), type=tracker.get_slot("response_type"), empathy_mode=empathy_mode)
        # print(text)
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        dispatcher.utter_message(text=text, buttons=buttons)
        return [SlotSet('pastResponses',updated_responses)]

class AskIfEventWasRecent(Action):
    def name(self) -> Text:
        return "action_is_recent"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        # Create a subset of EPRU dataset as measured by the similarity of user utterance to emotion utterances in the dataset for an emotion specified
        # dataset = generate_response_dataset(k=5, querry=tracker.get_slot('current_feeling'), emotion=tracker.get_slot('emotion'))
        # dataset.save_to_disk('./data/topKdataset')

        buttons = [{"title": "Да, эта ситуация произошла недавно", "payload": '/is_protocol_11_distressing'}, {"title": "Нет, это случилочь достаточно давно", "payload":'/is_protocol_6_distressing'}]
        empathy_mode = tracker.get_slot("empathy_mode")
        text = generate_next_response(prompt="Was the event recent?", past_sentences=tracker.get_slot('pastResponses'), type=tracker.get_slot("response_type"), novelty_weight=0.5, fluency_weight=0.25, empathy_weight=-0.1, empathy_mode=empathy_mode)
        if len(text.split(".")) == 1:
         text = tracker.get_slot("name") + " ," + text.lower()
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        dispatcher.utter_message(text=text, buttons=buttons)
        return [SlotSet('pastResponses',updated_responses)]

class Action6distressing(Action):
    def name(self) -> Text:
        return "action_is_protocol_6_distressing"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        empathy_mode = tracker.get_slot("empathy_mode")
        text = generate_next_response(prompt="Did you find protocol 6 distressing?", past_sentences=tracker.get_slot('pastResponses'), type=tracker.get_slot("response_type"), empathy_mode=empathy_mode)
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        buttons = [{"title": "Согласен, этот метод меня встревожил", "payload": '/ok_to_ask_more{"protocols_1":[13,7]}'}, {"title": "Нет, это упражнение не вызвало у меня волнения", "payload":'/ok_to_ask_more{"protocols_1":[6]}'}]
        dispatcher.utter_message(text=text, buttons=buttons, button_type="vertical")
        return [SlotSet('pastResponses',updated_responses)]

class Action6distressing(Action):
    def name(self) -> Text:
        return "action_is_protocol_11_distressing"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        empathy_mode = tracker.get_slot("empathy_mode")
        text = generate_next_response(prompt="Did you find protocol 11 distressing?", past_sentences=tracker.get_slot('pastResponses'), type=tracker.get_slot("response_type"), empathy_mode=empathy_mode)
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        buttons = [{"title": "Согласен, этот метод меня очень встревожил", "payload": '/ok_to_ask_more{"protocols_1":[7,8]}'}, {"title": "Нет, это упражнение не вызвало у меня волнения", "payload":'/ok_to_ask_more{"protocols_1":[11]}'}]
        dispatcher.utter_message(text=text, buttons=buttons, button_type="vertical")
        return [SlotSet('pastResponses',updated_responses)]

class AskMoreQuestions(Action):
    def name(self) -> Text:
        return "action_ok_to_ask_more"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        empathy_mode = tracker.get_slot("empathy_mode")
    
        text = generate_next_response(prompt="Is it ok to ask additional questions?", past_sentences=tracker.get_slot('pastResponses'), type=tracker.get_slot("response_type"), empathy_mode=empathy_mode)
        text = tracker.get_slot("name") + ", " + text.lower()
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        buttons = [{"title": "Я не против", "payload": '/additional_questions'}, {"title": "Нет, на сегодня достаточно вопросов", "payload":'/recommend_protocols{"positive_to_any_base_questions": "False"}'}]
        dispatcher.utter_message(text=text, buttons=buttons, button_type="vertical")

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
        empathy_mode = tracker.get_slot("empathy_mode")
        text = generate_next_response(prompt=prompt, past_sentences=tracker.get_slot('pastResponses'), type=tracker.get_slot("response_type"), empathy_mode=empathy_mode)
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        if len(available_prompts) == 1:
            buttons = [{"title": "Больше да, чем нет", "payload":'/recommend_protocols{"positive_to_any_base_questions": "True"}'}, {"title": "Думаю нет", "payload":'/recommend_protocols{"positive_to_any_base_questions": "False"}'}]
        else:
            buttons = [{"title": "Больше да, чем нет", "payload":'/recommend_protocols{"positive_to_any_base_questions": "True"}'}, {"title": "Нет", "payload":'/additional_questions'}]   
            available_prompts.remove(prompt)
        dispatcher.utter_message(text=text, buttons=buttons, button_type="vertical")

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
        print(protocols)
        
        protocols_map = {1: "Connecting with the Child" , 2: "Laughing at our Two Childhood Pictures" , 3: "Falling in Love with the Child" , 4: "Vow to Adopt the Child as Your Own Child", 5: "Maintaining a Loving Relationship with the Child", 6: "An exercise to Process the Painful Childhood Events", 7: "Protocols for Creating Zest for Life", 8: "Loosening Facial and Body Muscles", 9: "Protocols for Attachment and Love of Nature", 10: "Laughing at, and with One’s Self", 11: "Processing Current Negative Emotions", 12: "Continuous Laughter", 13: "Changing Our Perspective for Getting Over Negative Emotions", 14: "Protocols for Socializing the Child", 15: "Recognising and Controlling Narcissism and the Internal Persecutor",
        16: "Creating an Optimal Inner Model", 17:"Solving Personal Crises", 
        18: "Laughing at the Harmless Contradiction of Deep-Rooted Beliefs/Laughing at Trauma", 19:"Changing Ideological Frameworks for Creativity",
        20: "Affirmations" }

        if positive_to_any_base_questions == "True":
            protocols_additional_questions = prompt_mapping[tracker.get_slot('lastPrompt')]
            if protocols != None:
                protocols = protocols_additional_questions + protocols
            else:
                protocols = protocols_additional_questions

        if protocols == None:
            protocols = [1]
        buttons = []
        for protocol in protocols:
            button = dict()
            button["title"] = protocols_map[protocol]
            number = dict()
            number["number"] = protocol
            d = json.dumps(number)
            button["payload"] = f'/invite_to_protocol{d}'
            buttons.append(button)
        print(buttons)
        name = tracker.get_slot("name")
        dispatcher.utter_message(text = f"Спасибо, что уделили мне ваше ценное время, {name}! Исходя из нашей беседы, я подготовил несколько методик на ваше усмотрение.", buttons=buttons)
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
        buttons = [{"title": f"Лучше \U0001F601", "payload": '/respond_to_feedback{"response_to_feedback":"positive"}'}, {"title": f"Как и раньше \U0001F610", "payload": '/respond_to_feedback{"response_to_feedback":"encouraging_same"}'}, {"title": f"Хуже \U0001F612", "payload": '/respond_to_feedback{"response_to_feedback":"encouraging_worse"}'} ]
        dispatcher.utter_message(text=f"Спасибо, что нашли время и силы сделать это упражнение. Как вы сеебя ощущаете?", buttons=buttons)
        return [SlotSet("current_feeling", None), SlotSet("emotion_prediction", None), SlotSet("personality", None),SlotSet("emotion_prediction", None)]

class ActionRespondToFeedback(Action):
    def name(self) -> Text:
        return "action_respond_to_feedback"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        response_type = tracker.get_slot("response_to_feedback")
        print(response_type)
        if response_type == "positive":
            text = "Я счастлив, что ваше душевное состояние улучшшилось! С удовольствием могу предложить другую программу, если вы пожелаете."
        if response_type == "encouraging_same":
            text = "Я сожалею, что программа пока не повлияла на вас позитивно. Если пожелаете, я могу порекоммендовать другую программу."
        if response_type == "encouraging_worse":
            text = "Я сожалею, что программа пока не повлияла на вас позитивно. Если пожелаете, я могу порекоммендовать другую программу."
        
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

        protocols_map = {1: "Connecting with the Child" , 2: "Laughing at our Two Childhood Pictures" , 3: "Falling in Love with the Child" , 4: "Vow to Adopt the Child as Your Own Child", 5: "Maintaining a Loving Relationship with the Child", 6: "An exercise to Process the Painful Childhood Events", 7: "Protocols for Creating Zest for Life", 8: "Loosening Facial and Body Muscles", 9: "Protocols for Attachment and Love of Nature", 10: "Laughing at, and with One’s Self", 11: "Processing Current Negative Emotions", 12: "Continuous Laughter", 13: "Changing Our Perspective for Getting Over Negative Emotions", 14: "Protocols for Socializing the Child", 15: "Recognising and Controlling Narcissism and the Internal Persecutor",
        16: "Creating an Optimal Inner Model", 17:"Solving Personal Crises", 
        18: "Laughing at the Harmless Contradiction of Deep-Rooted Beliefs/Laughing at Trauma", 19:"Changing Ideological Frameworks for Creativity",
        20: "Affirmations" }

        if tracker.get_slot('emotion') != 'joy':
            protocols = tracker.get_slot("relevant_protocols") 
            buttons = []
            for protocol in protocols:
                button = dict()
                button["title"] = protocols_map[protocol]
                number = dict()
                number["number"] = protocol
                d = json.dumps(number)
                button["payload"] = f'/invite_to_protocol{d}'
                buttons.append(button)
            print(buttons)
            dispatcher.utter_message(text = f"Представляю на выбор несколько методик", buttons=buttons)
        else:
            protocols = tracker.get_slot("relevant_protocols") 
            print(protocols)
            buttons = []
            for protocol in protocols:
                button = dict()
                button["title"] = protocols_map[protocol]
                number = dict()
                number["number"] = protocol
                d = json.dumps(number)
                button["payload"] = f'/invite_to_protocol{d}'
                buttons.append(button)
            print(buttons)
            dispatcher.utter_message(text = f"Представляю на выбор несколько методик", buttons=buttons)

        return []

class ActionRecommendProtocolsForPositiveFeelings(Action):
    def name(self) -> Text:
        return "action_joy_enquire_for_protocol"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        # Creating a subset of responses to choose from
        if tracker.get_slot('personality') == 'Наташа' or tracker.get_slot('personality') == "Кирилл":
            dataset = generate_response_dataset(k=5, querry=tracker.get_slot('current_feeling'), emotion=tracker.get_slot('emotion'))
            dataset.save_to_disk('./data/topKdataset')
        else:
            synthetic_dataset = generate_synthetic_dataset(k=10,emotion=tracker.get_slot('emotion'))
            synthetic_dataset.save_to_disk('./data/randomSyntheticDataset')

        prompt = "Would you like to attempt protocols for joy?"
        type = tracker.get_slot("response_type")
        print(type)
        empathy_mode = tracker.get_slot("empathy_mode")
        text = generate_next_response(prompt=prompt, past_sentences=tracker.get_slot('pastResponses'), type=tracker.get_slot("response_type"), empathy_mode=empathy_mode)

        buttons = []
        yes_button = dict()
        yes_button["title"] = "Я не против!"
        no_button = dict()
        no_button['title'] = "Не сегодня"
        yes_button["payload"] = '/protocol_recommendation_follow_up'
        no_button["payload"] = '/goodbye'

        buttons.extend([yes_button, no_button]) 
        dispatcher.utter_message(text=text, buttons=buttons)



        return [SlotSet('relevant_protocols', [9, 10, 11])]



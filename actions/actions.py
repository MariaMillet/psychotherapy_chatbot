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

import datasets
from datasets import load_dataset
from datasets import DatasetDict
from datasets import Dataset

import pandas as pd
from numpy import random
from numpy.random.mtrand import randint

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

# def generate_synthetic_dataset(k=10, emotion='anger'):
#     dataSet = load_from_disk('./data/respSyntheticDataset')
#     subsetDataSet = dataSet.filter(lambda row: row['emotion']==emotion)
#     if emotion == "joy":
#         max_row = len(subsetDataSet["Would you like to attempt protocols for joy?"])
#     else:
#         max_row = len(subsetDataSet["Did you find protocol 6 distressing?"])
#     random_selection = subsetDataSet.select(random.choice(max_row, 10, replace=False))
#     print(random_selection)
#     return random_selection

def generate_synthetic_dataset(k=10, emotion='anger'):
    dataSet = load_from_disk('./data/respSyntheticDataset')
    subsetDataSet = dataSet.filter(lambda row: row['emotion']==emotion)

    if emotion == "joy":
        subsetDataSet = subsetDataSet.remove_columns(['emotion','Have you strongly felt or expressed any of the following emotions towards someone?', 'Do you believe you should be the savior of someone else?', 'Do you feel that you are trying to control someone?', 'Do you see yourself as the victim, blaming someone else for how negative you feel?', 'Are you always blaming and accusing yourself for when something goes wrong?', 'Is it possible that in previous conversations you may not have always considerdother viewpoints presented?', 'Are you undergoing a personal crisis (experience difficulties with loved ones e.g. falling out with friends)?', 'Was this emotion triggered by a specific event?', 'Was the event recent?', 'Did you find protocol 6 distressing?', 'Did you find protocol 11 distressing?', 'Is it ok to ask additional questions?'] )
        internal_dict = dict()
        max_int = len(subsetDataSet)
        # columns = subsetDataSet.column_names
        columns = [column for column in subsetDataSet.column_names if column[0:3]!="ppl" and column[0:3]!="emp"]
        print(columns)
        for column_name in columns:
            number_range = list(range(max_int))
            internal_dict[column_name] = []
            fluency_column = "ppl_"+column_name
            empathy_column = "emp_"+column_name
            internal_dict[fluency_column] = []
            internal_dict[empathy_column] = []
            i = 0
            while i < 10:
                random_number = random.choice(number_range)
                number_range.remove(random_number)
                utterance = subsetDataSet[column_name][random_number]
                emp_score = subsetDataSet[empathy_column][random_number]
                fluency_score = subsetDataSet[fluency_column][random_number]
                if not pd.isna(utterance):
                    i += 1
                    internal_dict[column_name].append(utterance)
                    internal_dict[fluency_column].append(fluency_score)
                    internal_dict[empathy_column].append(emp_score)
    else:
        subsetDataSet = subsetDataSet.remove_columns(['Would you like to attempt protocols for joy?', 'emotion'] )
        internal_dict = dict()
        max_int = len(subsetDataSet)
        
        # columns = subsetDataSet.column_names
        columns = [column for column in subsetDataSet.column_names if column[0:3]!="ppl" and column[0:3]!="emp"]
        print(columns)
        for column_name in columns:
            number_range = list(range(max_int))
            internal_dict[column_name] = []
            fluency_column = "ppl_"+column_name
            empathy_column = "emp_"+column_name
            internal_dict[fluency_column] = []
            internal_dict[empathy_column] = []
            i = 0
            while i < 10:
                random_number = random.choice(number_range)
                number_range.remove(random_number)
                utterance = subsetDataSet[column_name][random_number]
                emp_score = subsetDataSet[empathy_column][random_number]
                fluency_score = subsetDataSet[fluency_column][random_number]
                if not pd.isna(utterance):
                    i += 1
                    internal_dict[column_name].append(utterance)
                    internal_dict[fluency_column].append(fluency_score)
                    internal_dict[empathy_column].append(emp_score)


    # synthetic_df_columns = subsetDataSet.column_names
    synthetic_df_columns = internal_dict.keys()
    print(synthetic_df_columns)
    synthetic_df = pd.DataFrame(columns=synthetic_df_columns)
    df = pd.DataFrame.from_dict(internal_dict)
    synthetic_df = pd.concat([synthetic_df, df])
    dataset = Dataset.from_pandas(synthetic_df)
    return dataset


def generate_next_response(prompt, past_sentences, type, novelty_weight=0.5, fluency_weight=0.25, empathy_weight=0.25, empathy_mode="medium"):
  if type=="synthetic":
    dataset = load_from_disk('./data/randomSyntheticDataset')
    # print(dataset[0]["Would you like to attempt protocols for joy?"])
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
        # buttons = [{"title": "Кирилл - выражает свои мысли ясно, логично и предельно кратко \U0001F913", "payload": '/personality{"personality":"Кирилл", "response_type":"human", "empathy_mode":"medium"}'}, {"title": "Наташа - сама доброта, общается максимаально добро и ласково \U0001F60D ", "payload": '/personality{"personality":"Наташа", "response_type":"human", "empathy_mode":"high"}'}, {"title": "Компьюша - наиболее непредскауемый робот и наболее креативный в генерации ответов. Не судите его  строго, если иногда у него не всё получается \U0001F910 ", "payload": '/personality{"personality":"Компьюша", "response_type":"synthetic", "empathy_mode":"high"}'}]
        buttons = [{"title": "Кирилл - выражает свои мысли ясно, логично и предельно кратко \U0001F913", "payload": '/personality{"personality":"Кирилл"}'}, {"title": "Наташа - сама доброта, общается максимаально добро и ласково \U0001F60D ", "payload": '/personality{"personality":"Наташа"}'}, {"title": "Компьюша - наиболее непредскауемый робот и наболее креативный в генерации ответов. Не судите его  строго, если иногда у него не всё получается \U0001F910 ", "payload": '/personality{"personality":"Компьюша"}'}]
        dispatcher.utter_message(text=f"Пожалуста, выберите с кем из наших умных и человечных \U0001F607 психологов вы бы хотели пообщаться", buttons=buttons)
        return []

class AskForSlotActionFeeling(Action):
    def name(self) -> Text:
        return "action_ask_name"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        psychologist = tracker.get_slot("personality")
        if psychologist == "Кирилл":
            response_type = "human"
            empathy_mode = "medium"
        elif psychologist == "Наташа":
            response_type = "human"
            empathy_mode = "high"
        else:
            response_type = "synthetic"
            empathy_mode = "high"
        dispatcher.utter_message(text=f"Добрый день, я виртуальный психолог, меня зовут {psychologist}. Как я могу к вам обращаться?")
        return [SlotSet("response_type", response_type), SlotSet("empathy_mode", empathy_mode)]

class AskForSlotActionFeeling(Action):
    def name(self) -> Text:
        return "action_ask_current_feeling"


    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        if len(tracker.get_slot("name").split())!=1:
            dispatcher.utter_message(text="Извините, я пока не могу понять имя, длиннее одного слова")
            name = "милый друг"
        else:
            name = tracker.get_slot("name").capitalize()
        dispatcher.utter_message(text=f"Очень приятно, {name}! Я практикую методику SAT, и сделаю всё, чтобы улучшить ваше душевное состояние. Для начала, опишите, пожалуйста, как вы себя ощущаете? \u270F\uFE0F ")
        # dispatcher.utter_message(text=f"Please select who you want to speak with \u270F\uFE0F", buttons=buttons)
        # buttons = [{"title": "yes", "payload": '/joy_enquire_for_protocol{"emotion":"joy"}'}, {"title": "Нет, вы не угадали мое настроение", "payload":'/not_correct_prediction'}]
        # prompts = ['Have you strongly felt or expressed any of the following emotions towards someone?', 'Do you believe you should be the savior of someone else?', 'Do you see yourself as the victim, blaming someone else for how negative you feel?',
        #     'Do you feel that you are trying to control someone?', 'Are you always blaming and accusing yourself for when something goes wrong?', 'Is it possible that in previous conversations you may not have always considerdother viewpoints presented?',
        #     'Are you undergoing a personal crisis (experience difficulties with loved ones e.g. falling out with friends)?']
        prompts = ['Do you believe you should be the savior of someone else?', 'Do you see yourself as the victim, blaming someone else for how negative you feel?',
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

        # Create a subset of EPRU dataset as measured by the similarity of user utterance to emotion utterances in the dataset for an emotion specified
        if tracker.get_slot('personality') == "Компьюша":
            synthetic_dataset = generate_synthetic_dataset(k=10,emotion=tracker.get_slot('emotion'))
            print(synthetic_dataset)
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

        buttons = [{"title": "Эта ситуация произошла недавно", "payload": '/is_protocol_11_distressing'}, {"title": "Это случилочь достаточно давно", "payload":'/is_protocol_6_distressing'}]
        empathy_mode = tracker.get_slot("empathy_mode")
        text = generate_next_response(prompt="Was the event recent?", past_sentences=tracker.get_slot('pastResponses'), type=tracker.get_slot("response_type"), novelty_weight=0.5, fluency_weight=0.25, empathy_weight=-0.1, empathy_mode=empathy_mode)
        if len(text.split(".")) == 1:
         text = tracker.get_slot("name").capitalize() + ", " + text.lower()
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
        buttons = [{"title": "Да, этот метод меня встревожил", "payload": '/ok_to_ask_more{"protocols_1":[13,7]}'}, {"title": "Нет, это упражнение не вызвало у меня волнения", "payload":'/ok_to_ask_more{"protocols_1":[6]}'},
        {"title": "Я ещё не делал(а) это упражнение 🤔", "payload":'/ok_to_ask_more{"protocols_1":[6]}'}]
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
        buttons = [{"title": "Да, этот метод меня очень встревожил", "payload": '/ok_to_ask_more{"protocols_1":[7,8]}'}, {"title": "Нет, это упражнение не вызвало у меня волнения", "payload":'/ok_to_ask_more{"protocols_1":[11]}'},
        {"title": "Я ещё не делал(а) это упражнение 🤔", "payload":'/ok_to_ask_more{"protocols_1":[11]}'}]
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
        text = tracker.get_slot("name").capitalize() + ", " + text.lower()
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        buttons = [{"title": "Я не против 🙌", "payload": '/additional_questions'}, {"title": "Нет, на сегодня достаточно вопросов 🫢", "payload":'/recommend_protocols{"positive_to_any_base_questions": "False"}'}]
        dispatcher.utter_message(text=text, buttons=buttons, button_type="vertical")

        return [SlotSet('pastResponses',updated_responses)]

class AdditionalQuestions(Action):
    def name(self) -> Text:
        return "action_additional_questions"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        available_prompts = tracker.get_slot("additionalQuestionsPrompts")
        prompt = random.choice(available_prompts)
        empathy_mode = tracker.get_slot("empathy_mode")

        # new
        additional_question_number = tracker.get_slot("additional_question_number")
        if tracker.get_slot('response_type') !=  "human":
            if additional_question_number % 3 != 0:
                text = generate_next_response(prompt=prompt, past_sentences=tracker.get_slot('pastResponses'), type=tracker.get_slot("response_type"), empathy_mode="low")
                if len(text.split('.',1)) > 1:
                    text = text.split('.',1)[1]
            else:
                text = generate_next_response(prompt=prompt, past_sentences=tracker.get_slot('pastResponses'), type=tracker.get_slot("response_type"), empathy_mode=empathy_mode)

        else:
            if additional_question_number % 2 != 0:
                text = generate_next_response(prompt=prompt, past_sentences=tracker.get_slot('pastResponses'), type=tracker.get_slot("response_type"), empathy_mode="low")
                if len(text.split('.',1)) > 1:
                    text = text.split('.',1)[1]
            else:
                text = generate_next_response(prompt=prompt, past_sentences=tracker.get_slot('pastResponses'), type=tracker.get_slot("response_type"), empathy_mode=empathy_mode)
        if text[-1] != "?":
            text = text + "?"
        updated_responses = add_response_to_past_responses(latest_response=text, past_responses=tracker.get_slot('pastResponses'))
        if len(available_prompts) == 1:
            buttons = [{"title": "Больше да, чем нет", "payload":'/recommend_protocols{"positive_to_any_base_questions": "True"}'}, {"title": "Думаю нет", "payload":'/recommend_protocols{"positive_to_any_base_questions": "False"}'}]
        else:
            buttons = [{"title": "Больше да, чем нет", "payload":'/recommend_protocols{"positive_to_any_base_questions": "True"}'}, {"title": "Нет", "payload":'/additional_questions'}]   
            available_prompts.remove(prompt)
        dispatcher.utter_message(text=text.capitalize(), buttons=buttons, button_type="vertical")

        return [SlotSet("additionalQuestionsPrompts", available_prompts), SlotSet('pastResponses',updated_responses), SlotSet('lastPrompt', prompt), SlotSet('additional_question_number', additional_question_number+1)] 


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
        
        protocols_map = {1: "1. Connecting with the Child" , 
        2: "2. Laughing at our Two Childhood Pictures" , 
        3: "3. Falling in Love with the Child" , 
        4: "4. Vow to Adopt the Child as Your Own Child", 
        5: "5. Maintaining a Loving Relationship with the Child", 
        6: "6. An exercise to Process the Painful Childhood Events", 
        7: "7. Protocols for Creating Zest for Life", 
        8: "8. Loosening Facial and Body Muscles", 
        9: "9. Protocols for Attachment and Love of Nature", 
        10: "10. Laughing at, and with One’s Self", 
        11: "11. Processing Current Negative Emotions", 
        12: "12. Continuous Laughter", 
        13: "13. Changing Our Perspective for Getting Over Negative Emotions", 
        14: "14. Protocols for Socializing the Child", 
        15: "15. Recognising and Controlling Narcissism and the Internal Persecutor",
        16: "16. Creating an Optimal Inner Model", 
        17:"17. Solving Personal Crises", 
        18: "18. Laughing at the Harmless Contradiction of Deep-Rooted Beliefs/Laughing at Trauma", 
        19:"19. Changing Ideological Frameworks for Creativity",
        20: "20. Affirmations" }

        if positive_to_any_base_questions == "True":
            protocols_additional_questions = prompt_mapping[tracker.get_slot('lastPrompt')]
            if protocols != None:
                protocols = protocols_additional_questions + protocols
            else:
                protocols = protocols_additional_questions
        else:
            if protocols != None:
                protocols = [13] + protocols
            else:
                protocols = [13]

        # if protocols == None:
        #     protocols = [1]
        buttons = []
        protocols = sorted(list(set(protocols)))
        print(f"here are {protocols}")
        for protocol in protocols:
            button = dict()
            button["title"] = protocols_map[protocol]
            number = dict()
            number["number"] = protocol
            d = json.dumps(number)
            button["payload"] = f'/invite_to_protocol{d}'
            buttons.append(button)
        print(buttons)
        name = tracker.get_slot("name").capitalize()
        dispatcher.utter_message(text = f"Спасибо, что уделили мне ваше ценное время, {name}! Исходя из нашей беседы, я подготовил несколько методик на ваше усмотрение.", buttons=buttons)
        return [SlotSet('relevant_protocols', protocols)]

class ActionInviteToProtocols(Action):
    def name(self) -> Text:
        return "action_invite_to_protocol"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        protocols_map = {1: "1. Connecting with the Child" , 
        2: "2. Laughing at our Two Childhood Pictures" , 
        3: "3. Falling in Love with the Child" , 
        4: "4. Vow to Adopt the Child as Your Own Child", 
        5: "5. Maintaining a Loving Relationship with the Child", 
        6: """6. An exercise to Process the Painful Childhood Events
            With closed eyes, recall a painful scene from childhood e.g. emotional or physical abuse in as much detail as possible, 
            and associate the face of the child you were with your unhappy photo. After recalling this event and the related emotions, imagine your adult self approaching and embracing the child like
            a parent embracing a child in distress.
            While your eyes are still closed, continue to imagine supporting and cuddling the child, loudly supporting them (Examples: “Why are you hitting my child?” and “My darling, I will not let them hurt you any more.”). 
            Massage your face while doing so, which we interpret as cuddling the child..""",

        7: """                      7. Protocols for Creating Zest for Life
        Using a mirror, imagine the reflection is your childhood self and loudly recite to it your selected happy love songs, using your entire body. 
        Repeat songs and poems in many different circumstances e.g. while walking on the street or doing housework, to be able to integrate them into your life.""", 
        8: """                      8. Loosening Facial and Body Muscles
           You should loosen your muscles at least twice a day as you sing with your face and entire body, as if playing, dancing, laughing and having fun with the child as parents do with children.""", 
        9: """                      9. Protocols for Attachment and Love of Nature
            To create an attachment with nature, you should visit a park or forest and spend time admiring nature, e.g. admiring a beautiful tree, as if seeing its branches and leaves for the first time. 
            Repeat continuously and with different trees until you feel you have formed an attachment with nature. 
            This will help to modulate your emotions and you will want to spend more time with nature each day.""", 
        10: """                     Method 10. Laughing at, and with One’s Self
            Begin laughing with yourself about a small accomplishment e.g. in sports, housework, or any other task, however small or unimportant. 
            With every small accomplishment, you should smile as if victorious, and gradually change this smile to laughter, and make this laughter last longer and longer. 
            By practising this you will be able to smile and laugh without ridicule about anything you have said or done in the past while maintaining compassion for your childhood self.""", 
        11: """                     11. Processing Current Negative Emotions
            With closed eyes, imagine the unhappy photo and project the unhappy emotions, e.g. anger, sorrow, towards the photo that represents the child. 
            As with Type 6, we make contact with our adult self to attend to and care for the child to support the child and modulate the child’s negative emotions.
            While projecting these negative emotions, loudly reassure the child and massage your own face, which we interpret as cuddling the child. 
            Continue this until you have contained the negative emotions, at which point you can switch to focusing on the happy photo.""", 
        12: """12. Continuous Laughter
            At a time when you are alone, you should open your mouth slightly, loosen your face muscles, form a Duchenne smile and slowly repeat one of the following phrases as if laughing: eh, eh, eh, eh; ah, ah, ah, ah; oh, oh, oh, oh; uh, uh, uh, uh; or ye, ye, ye, ye.
            If a subject is needed for laughter, you can think about the silliness of the exercise. This exercise is a good antidote for stress.""", 
        13: """13. Changing Our Perspective for Getting Over Negative Emotions
            To break free of the gravitational field of powerful negative patterns that emerge when we are stuck in the storeroom of negative emotions, or the “psychological abyss”, stare at the black vase in the Gestalt vase picture (below). When you see the white faces, laugh out loud.
            Having created a positive powerful pattern of love with the child through previous exercises, you can now depart from the field of negative patterns by singing your happy love song to enter the gravitational field of love for the child instead.
            This is like changing our interpretation of the above image and instead of seeing a black vase of negative emotions discovering two white faces, you see the child and the adult self who are now looking at each other.""", 
        14: """14. Protocols for Socializing the Child
                By repeating protocols 1-13 you can reduce negative emotions and increase positive affects. 
                You should gradually be able to perform these exercises with eyes open and can integrate them into your daily life. You should be able to extend compassion for the child to other people. The adult self should become aware of any narcissistic tendencies or anti-social feelings of the child e.g. envy, jealousy, greed, hatred, mistrust, malevolence, controlling behavior and revengefulness.
                The adult self can behave like a parent to contain these emotions and discourage anti-social feelings and attitudes of the child by expressing affection to the child and simulating cuddles by massaging your face.
                The adult self should try to direct the child’s anger and negative energy towards playing, creativity and development. As the child’s positive affects increase and his/her negative affects decrease, by expressing positive emotions he/she can attract more positive reactions from others, and in turn gain a more positive outlook toward others.""", 
        15: """15. Recognising and Controlling Narcissism and the Internal Persecutor
                The adult self becomes aware of the facets of the trauma triangle: internal persecutor, victim, and rescuer. The adult self examines the effects of the triangle (narcissism, lack of creativity) in daily life and previous experiences.
                Your adult self should then review an important life experience and your social and political views as an adult, with awareness of how the internal persecutor operates. Your adult self should then create a list of examples from own experiences about how the internal persecutor operates, and carefully analyse these for examples of being drawn to trauma, being traumatized by the internal persecutor, and projecting the internal persecutor.
                You should be able to then re-evaluate your own experiences, control the internal persecutor and narcissism and be able to develop creativity.""",
        16: """16. Creating an Optimal Inner Model
                With awareness of the internal persecutor, we will recognise emotions of the child that were learned from parents or through interactions with them. With the guidance of the adult self, who can transfer
                compassion for the child to others, the child will learn to avoid projecting the internal persecutor (which would lead to them becoming the victim or rescuer).""", 
        17:"""17. Solving Personal Crises
                As you continue to practice the protocol for modulating negative affects and the protocol for laughter, ask your child the following:
                • How can you see the crisis as a way of becoming stronger? (ha ha ha)
                • How can you interpret the crisis as a way of reaching your high goal? (ha ha ha)
                • Has the internal persecutor been projecting onto others again?
                The adult self asks the following questions:
                • What is the similarity between this crisis and ones I have faced before?
                • How is it similar to the family crisis I experienced as a child?
                • Aren’t the other person’s positive attributes greater than his/her negative ones?
                • How would a mature person interpret the crisis in comparison to my child?
                • Can I see it from the perspective of someone else?
                • Can I put myself in their place and understand their affects?
                • Given my new inner working model can I find a way to calm the people involved in the crisis so we can find a better solution for it?
                • If not, can I respectfully maintain my distance and end the argument?""", 
        18: """18. Laughing at the Harmless Contradiction of Deep-Rooted Beliefs/Laughing at Trauma
        (i): Laughing at the harmless contradiction of deep-rooted beliefs
            “To those human beings who are of any concern to me I wish suffering, desolation, sickness, ill- treatment, indignities—I wish that they should not remain unfamiliar with profound self-contempt, the torture of self-mistrust, the wretchedness of the vanquished: I have no pity for them, because I wish them the only thing that can prove today whether one is worth anything or not—that one endures.”
            This is meaningful with, “What doesn’t kill me makes me stronger.” Nietzsche’s wish is funny and a harmless contradiction of our deep-rooted beliefs. As we read the quote above, we remember our past sufferings and begin to laugh out loud when we get to “...I wish suffering...”
            (i) continued: Laughing at trauma
            First, visualize a painful event that took place in the distant past that you have struggled with for a long time, and despite its painfulness try to see a positive impact it has had. We start with a painful event that happened in the distant past, so that by now we have been able to adjust our negative affects toward it. After repeated daily exercises, once we have experienced the forceful effectiveness of laughing at distant problems, we can gradually begin to laugh at more recent painful memories.
            (ii): Laughing at trauma
            In expectation of hearing a funny joke we loosen our facial muscles, slightly open our mouths, and to grasp the incongruity in the joke we move our eyebrows up as a sign of surprise. As we repeat the sentences out loud, we slowly begin to laugh as we wait for the second part. And once we get to the first sentence of the second part, which is in complete contrast to our beliefs, we laugh out loud.
            Not only should you: bear it, accept it, try to deal with it, tolerate its memory, try harder to endure its memory, adapt yourself to its memory, analyze and understand it and by doing so modulate your negative emotions and learn lessons for the future, try to soften your thoughts, depressive emotions, and anxieties, try to ...
            Like Nietzsche’s wish consider it a cherished treasure (ha ha ha...), treasure it with great love (ha ha ha...), welcome its challenges with all your heart (ha ha ha...), consider it a good omen with all your heart (ha ha ha...), consider its challenges a great fortune (ha ha ha...), celebrate its memory (ha ha ha...), celebrate its memory with great joy (ha ha ha...), consider it a true love (ha ha ha...), consider it a true love with great passion and intimacy (ha ha ha...) ...
            After repeated practice of the laughing exercises you can begin to apply it to things that worry you in the present and the future.""", 
        19:"""19. Changing Ideological Frameworks for Creativity
            We challenge our usual ideological framework to weaken one-sided patterns and encourage spontaneity and the examination of issues from multiple perspectives. Practice with subjects that you have deep- rooted beliefs and are excited about e.g. anything from political/social issues to ideas on marriage and sexuality. For instance, examine the topic of racism and consider whether you have any latent racism and consider this subject in the dual role of proponent and opponent.
            Repeat with topics where you may have stronger views e.g. marriage and sexual orientation. If you are politically in the center, consider the subject both from a leftist and rightist point of view and try to understand both sides of the issue and see the subject from three perspectives.""",
        20: """20. Affirmations
        
        Put together a list of instructive sayings by different important figures. Choose ones that have an impact on you from the start and can provide you with strength in the long path for reaching your ultimate goal. Read them out loud.
        A few examples:
        • “My formula for greatness in a human being is Amor Fati: that one wants nothing to be other than it is, not in the future, not in the past, not in all eternity.” (Nietzsche)
        • “I assess the power of a will by how much resistance, pain, torture it endures and knows how to turn it to its advantage.” (Nietzsche)
        • Life is not easy. At times we inevitably suffer from hopelessness and paranoia unless if we have an ideal goal that helps us surpass suffering, weakness, and betrayals.” (Bronstein)""" }

        protocols_instructions = protocols_map[tracker.get_slot("number")]
        dispatcher.utter_message(text=protocols_instructions)
        # data = {"payload":"pdf_attachment", "title": "PDF Title", "url": "https://drive.google.com/file/d/1TfdZQQ8bI4WIPPWLyDFond71RER9hFFJ/view?usp=sharing"}
        # dispatcher.utter_message(json_message=data)
       
        


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
        buttons_negative = [{"title": f"Лучше \U0001F601", "payload": '/respond_to_feedback{"response_to_feedback":"positive"}'}, {"title": f"Как и раньше \U0001F610", "payload": '/respond_to_feedback{"response_to_feedback":"encouraging_same"}'}, {"title": f"Хуже \U0001F612", "payload": '/respond_to_feedback{"response_to_feedback":"encouraging_worse"}'} ]
        buttons_positive = [{"title": f"Ещё Лучше 😁", "payload": '/respond_to_feedback{"response_to_feedback":"positive"}'}, {"title": f"По-прежнему хорошо 😊 ", "payload": '/respond_to_feedback{"response_to_feedback":"encouraging_same"}'}, {"title": f"Хуже \U0001F612", "payload": '/respond_to_feedback{"response_to_feedback":"encouraging_worse"}'} ]
        if tracker.get_slot("emotion") == "joy":
            dispatcher.utter_message(text=f"Спасибо, что нашли время и силы сделать это упражнение. Как вы себя ощущаете?", buttons=buttons_positive)
        else:
            dispatcher.utter_message(text=f"Спасибо, что нашли время и силы сделать это упражнение. Как вы себя ощущаете?", buttons=buttons_negative)
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
            text = "Я счастлив, что ваше душевное состояние улучшилось 😊! С удовольствием могу предложить другую программу, если вы пожелаете."
        if response_type == "encouraging_same":
            if tracker.get_slot("emotion") == 'joy':
                text = "Прекрасно, что ваше сомочувствие осталось по-прежнему на высоте 🤗 Если пожелаете, я могу порекоммендовать другую программу."
            else:
                text = "Я сожалею, что программа пока не повлияла на вас позитивно 🥲. Если пожелаете, я могу порекоммендовать другую программу."
        if response_type == "encouraging_worse":
            text = "Я сожалею, что программа пока не повлияла на вас позитивно 🥲. Если пожелаете, я могу порекоммендовать другую программу."
        
        buttons = [{"title": "❌", "payload": '/end'}]

        yes_button = dict()
        yes_button["title"] = "✅"
        relevant_protocols = tracker.get_slot("relevant_protocols")
        if len(relevant_protocols) >= 1:
            yes_button["payload"] = '/protocol_recommendation_follow_up'
            recommendation_number = tracker.get_slot("recommendation_number") + 1
            print(f"number recom {recommendation_number}")
        else:
            yes_button["payload"] = '/greet'
            recommendation_number = 0

        buttons.append(yes_button) 

        # buttons = [{"title": "Yes", "payload": '/generate_response{"response_type":"positive"}'}, {"title": "No", "payload": '/generate_response{"response_type":"encouraging"}'}, {"title": "Worse", "payload": '/generate_response{"response_type":"encouraging"}'} ]
        dispatcher.utter_message(text=text, buttons=buttons)
        return [SlotSet("recommendation_number", recommendation_number)]
    
class ActionRecommendProtocols(Action):
    def name(self) -> Text:
        return "action_protocol_recommendation_follow_up"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        protocols_map = {1: "1. Connecting with the Child" , 2: "2. Laughing at our Two Childhood Pictures" , 3: "3. Falling in Love with the Child" , 
        4: "4. Vow to Adopt the Child as Your Own Child", 5: "5. Maintaining a Loving Relationship with the Child", 
        6: "6. An exercise to Process the Painful Childhood Events", 7: "7. Protocols for Creating Zest for Life", 8: "8. Loosening Facial and Body Muscles", 
        9: "9. Protocols for Attachment and Love of Nature", 10: "10. Laughing at, and with One’s Self", 11: "11. Processing Current Negative Emotions", 
        12: "12. Continuous Laughter", 13: "13. Changing Our Perspective for Getting Over Negative Emotions", 14: "14. Protocols for Socializing the Child", 
        15: "15. Recognising and Controlling Narcissism and the Internal Persecutor",
        16: "Creating an Optimal Inner Model", 17:"Solving Personal Crises", 
        18: "Laughing at the Harmless Contradiction of Deep-Rooted Beliefs/Laughing at Trauma", 19:"Changing Ideological Frameworks for Creativity",
        20: "Affirmations" }
        print(tracker.get_slot("recommendation_number"))
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
            possible_responses = ["Представляю на выбор несколько методик"]
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
            possible_responses = ["Представляю на выбор несколько методик"]
            dispatcher.utter_message(text = f"Представляю на выбор несколько методик", buttons=buttons)
            print(tracker.get_slot("recommendation_number"))
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
        yes_button["title"] = "Я не против! ✅"
        no_button = dict()
        no_button['title'] = "Не сегодня ❌"
        yes_button["payload"] = '/protocol_recommendation_follow_up'
        no_button["payload"] = '/end'

        buttons.extend([yes_button, no_button]) 
        dispatcher.utter_message(text=text, buttons=buttons)



        return [SlotSet('relevant_protocols', [9, 10, 11])]

class ActionEndConversation(Action):
    def name(self) -> Text:
        return "action_end"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        name = tracker.get_slot("name").capitalize()
        text = [f"{name}, спасибо за беседу, желаю вам всего наилучшего 💚!", f"{name}, спасибо за доверие и, надеюсь, до встречи 💙!"]
        rand_number = random.choice(len(text))
        text = text[rand_number]
        dispatcher.utter_message(text=text)

        text = f"Чтобы перезапустить чат напишите сначала '/restart', затем 'hi'"

        dispatcher.utter_message(text=text)

        return []



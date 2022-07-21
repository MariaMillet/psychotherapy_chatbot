from transformers import AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline

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
        prediction = max(preds, key=lambda x: x['score'])
        prediction_label = self.int2label(int(prediction['label'][-1]))
        return prediction_label
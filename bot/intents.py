import json


class Intent:

    def __repr__(self):

        return """Intent{%s}""" % self.__tag__

    def __init__(self, tag, patterns, responses, context, emotion_modifiers=None):

        self.__tag__       = tag
        self.__patterns__  = patterns
        self.__responses__ = responses
        self.__context__   = context
        self.__emotion_modifiers__ = emotion_modifiers if emotion_modifiers is not None else {
            'happiness': 1,
            'sadness': 0,
            'anger': 0,
            'fear': 0,
            'trust': 0,
            'disgust': 0,
            'anticipation': 0,
            'surprise': 0
        }

    def get_tag(self):

        return self.__tag__

    @property
    def tag(self):

        return self.get_tag()

    def get_patterns(self):

        return self.__patterns__

    @property
    def patterns(self):

        return self.get_patterns()

    def get_responses(self):

        return self.__responses__

    @property
    def responses(self):

        return self.get_responses()

    def get_context(self):

        return self.__context__

    @property
    def context(self):

        return self.get_context()

    def get_emotion_modifiers(self):

        return self.__emotion_modifiers__

    @property
    def emotion_modifiers(self):

        return self.get_emotion_modifiers()

    def __json__(self):

        return {'tag': self.get_tag(), 'patterns': self.get_patterns(), 'responses': self.get_responses(), 'context': self.get_context(), 'emotion_modifiers': self.__emotion_modifiers__}


class Intents:

    def __repr__(self):

        return f"""Intents[{', '.join([intent.__repr__() for intent in self.get_intents()])}]"""

    def __init__(self, intents):

        self.__intents__ = intents

    def get_intents(self):

        return self.__intents__

    @property
    def intents(self):

        return self.get_intents()

    def get_intent_by_tag(self, tag) -> Intent:

        for intent in self.get_intents():

            if intent.tag == tag:

                return intent

        return None

    def __json__(self):

        return {'intents': [intent.__json__() for intent in self.get_intents()]}

class IntentsLoader:

    @staticmethod
    def load(filename='intents.json', filetype=1):

        file = open(filename, 'r').read()
        j = json.loads(file)

        return j

    @staticmethod
    def loadintents(filename='intents.json', filetype=1):

        intents = IntentsLoader.load(filename, filetype)
        ints = []

        for intent in intents['intents']:

            ints.append(
                Intent(intent['tag'], intent['patterns'], intent['responses'], intent['context'], intent['emotion_modifiers'])
            )

        return Intents(ints)
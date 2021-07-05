import pickle
import os
import bot.intents as intents
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tflearn
from tensorflow.python.framework import ops
import numpy
import logging
import random
from bot.entry import Entry as ConsoleEntry
from tkinter import *
from _thread import start_new_thread

# Console

class DeveloperConsole:

    consoles = []

    @staticmethod
    def console_log(bot, entry):

        for console in DeveloperConsole.consoles:

            developer_chat_window = console.developer_chat_window

            message = f'{bot.name} @ console  ' + entry.message.rstrip().lstrip() + '\n'

            developer_chat_window.config(state=NORMAL)
            developer_chat_window.insert(END, message, 'console_log')
            developer_chat_window.config(state=DISABLED)
            developer_chat_window.see(END)

    def __call__(self, bot, entry):

        DeveloperConsole.console_log(bot, entry)

    def __init__(self, root, dimensions):

        DeveloperConsole.consoles.append(self)

        self.root = root
        self.frame = Frame(self.root, padx=5, pady=5)
        self.w, self.h = dimensions
        self.developer_chat_window = Text(self.frame, bd=1, bg='gray18')
        self.developer_chat_window.config(state=DISABLED)
        self.developer_chat_window.tag_config('console_log', foreground='lime green')
        self.frame.place(x=5, y=5, width=980, height=400)
        self.developer_chat_window.pack(fill=BOTH)
        self.frame.pack(fill=BOTH)

    def show(self):

        self.frame.tkraise()

    def hide(self):

        pass

class Options:

    default_developer_console = None

stemmer = LancasterStemmer()
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Bot utils

def make_dirs(filename):

    if not os.path.exists(path := os.path.dirname(filename)):

        os.makedirs(path)


def bag_of_words(s, words):

    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s_word in s_words:

        for i, w in enumerate(words):

            if w == s_word:

                bag[i] = 1

    return numpy.array(bag)


def combine_intents(intentsx):

    remade = {'intents': []}

    for intent in intentsx['intents']:

        intent_tag = {}

        for key, val in intent.items():

            intent_tag[key] = val

        match = False

        for made in remade['intents']:

            if intent_tag['tag'] == made['tag']:

                for pattern in intent_tag['patterns']:

                    if pattern not in made['patterns']:

                        made['patterns'].append(pattern)

                for response in intent_tag['responses']:

                    if response not in made['responses']:

                        made['responses'].append(response)

                match = True

        if not match:

            remade['intents'].append(intent_tag)

    return remade

# Bot class


class Bot:

    def __import_intents__(self):

        self.console_log('Importing bot intents')

        # Attempt to import intents

        intents_path = f'bot/personality/{self.name}/{self.name}_intents.json'

        try:

            self.console_log(f'Attempting to convert intents path: {intents_path} to proper intents')

            self.intents = intents.IntentsLoader.loadintents(intents_path)

            self.console_log('Intent path properly loaded')

        except FileNotFoundError:

            self.console_log('Intent path could not be loaded, creating intent path')

            make_dirs(intents_path)

            with open(intents_path, 'w') as file:

                starting_intents = {"context": "", "emotion_modifiers": {"anger": 0, "anticipation": 5, "disgust": 0, "fear": 0, "happiness": 0.1, "sadness": 0, "surprise": 10, "trust": 0}, "patterns": ["Hello", "Hi", "Yo", "How are ya"], "responses": ["Hello again!", "Hi!", "How are you?"], "tag": "personal.greeting"}
                file.write(json.dumps({"intents": [starting_intents]}))

                self.console_log('Intent now uploaded with starting intents')

            self.intents = intents.IntentsLoader.loadintents(intents_path)

    def __export_intents__(self, update=None):

        self.console_log('Exporting intents')

        current_intents = self.intents.__json__()
        current_intents = combine_intents(current_intents)

        if update:

            self.console_log(f'Adding new intents {update}')

            current_intents['intents'].append(update)

        intents_path = f'bot/personality/{self.name}/{self.name}_intents.json'

        try:

            self.console_log(f'Attempting to write out intents to path {intents_path}')

            with open(intents_path, 'w') as file:

                file.write(json.dumps(current_intents, sort_keys=True, indent=4))

            self.console_log('Successfully wrote out intents path')

        except FileNotFoundError:

            self.console_log('Could not write out intents path, retrying sequence: I.I, E.I')

            self.__import_intents__()
            self.__export_intents__()

    def __assert_pickle_data__(self):

        self.console_log('Attempting to assert pickle data')

        pickle_path = f'bot/personality/{self.name}/{self.name}_data_pickle.pickle'

        make_dirs(pickle_path)

        # Create the data pickle

        self.words, self.labels, docx, docy = [], [], [], []

        for intent in self.intents.get_intents():

            for pattern in intent.get_patterns():

                wds = nltk.word_tokenize(pattern)
                self.words.extend(wds)
                docx.append(wds)
                docy.append(intent.get_tag())

            if (tag := intent.tag) not in self.labels:

                self.labels.append(tag)

        self.words = [stemmer.stem(w.lower()) for w in self.words if w not in ('?',)]
        self.words = sorted(list(set(self.words)))
        self.labels = sorted(self.labels)

        out_empty = [0 for _ in range(len(self.labels))]

        self.training, self.output = [], []

        for x, doc in enumerate(docx):

            bag, wds = [], [stemmer.stem(w) for w in doc]

            for w in self.words:

                if w in wds:

                    bag.append(1)

                else:

                    bag.append(0)

            output_row = out_empty[:]
            output_row[self.labels.index(docy[x])] = 1

            self.training.append(bag)
            self.output.append(output_row)

        self.training, self.output = numpy.array(self.training), numpy.array(self.output)

        with open(pickle_path, 'wb') as file:

            pickle.dump((self.words, self.labels, self.training, self.output), file)

    def __import_data_pickle__(self):

        self.console_log('Attempting to import pickle data')

        pickle_path = f'bot/personality/{self.name}/{self.name}_data_pickle.pickle'

        try:

            self.console_log(f'Attempting to open pickle of path {pickle_path}')

            with open(pickle_path, 'rb') as file:

                self.words, self.labels, self.training, self.output = pickle.load(file)

            self.console_log('Successfully wrote out pickle data binary')

        except FileNotFoundError:

            self.console_log('Could not write out pickle data, retrying')

            self.__assert_pickle_data__()
            self.__import_data_pickle__()

        except ValueError:

            self.console_log('Unexpected pickle data unpacking error')
            raise Exception(f'When unpacking data pickle for {self.name}, unpacking problem occurred')

    def __create_tflearn_model__(self):

        self.console_log('Creating tflearn model')

        ops.reset_default_graph()

        self.net = tflearn.input_data(shape=[None, len(self.training[0])])

        self.net = tflearn.fully_connected(self.net, 8)
        self.net = tflearn.fully_connected(self.net, 8)
        self.net = tflearn.fully_connected(self.net, 8)

        self.net = tflearn.fully_connected(self.net, len(self.output[0]), activation='softmax')
        self.net = tflearn.regression(self.net)

        self.model = None
        self.model = tflearn.DNN(self.net)

        self.console_log('Successfully created model, attempting to save')

    def __refit_tflearn_model__(self, epochs=1000, progress=True, tracersp: [object, ] = (), tracerse: [object, ] = ()):

        self.console_log(f'Attempting to refit tflearn model with {epochs} epochs')

        if self.model is None:

            self.console_log('Model not found, creating')

            self.__create_tflearn_model__()

        model_path = f'bot/personality/{self.name}/{self.name}_bot_model.tflearn'

        try:

            self.console_log(f'Trying to load model using path {model_path}')
            self.model.load(model_path)
            self.console_log('Successfully loaded model')

        except FileNotFoundError:

            self.console_log('Could not load model, creating model now')
            make_dirs(model_path)
            self.__create_tflearn_model__()
            self.__refit_tflearn_model__(epochs, progress, tracersp, tracerse)

        self.__training_process__ = 0
        self.__training_epochs__ = epochs
        for _ in tracersp: _ = 0
        for _ in tracerse: _ = epochs

        for _ in range(epochs):

            self.model.fit(self.training, self.output, n_epoch=1, batch_size=8)

            self.__training_process__ += 1

            for _ in tracersp: _ += 1

            self.console_log('Completed single epoch using batch_size=8')

        for _ in tracersp: _ = 0
        for _ in tracerse: _ = epochs

    def __save_tflearn_model__(self):

        self.console_log('Attempting to save tflearn model..')

        model_path = f'bot/personality/{self.name}/{self.name}_bot_model.tflearn'

        if self.model is None:

            self.console_log('Tflearn model does not exist, creating')

            self.__create_tflearn_model__()

        self.console_log('Attempting to saved tflearn model')

        self.model.save(model_path)

        self.console_log('Successfully saved tflearn model')

    def __load_tflearn_model__(self):

        self.console_log('Attempting to hard load tflearn model')

        model_path = f'bot/personality/{self.name}/{self.name}_bot_model.tflearn'

        if self.model is None:

            self.__create_tflearn_model__()

        try:

            self.model.load(model_path)

            return True

        except FileNotFoundError:

            self.__save_tflearn_model__()

    def __reload_intents__(self):

        self.console_log('Semi-Fullstack, exporting and importing intents')

        self.__export_intents__()
        self.__reload_intents__()

    def __start__(self):

        self.console_log('Called __start__() in __start__()')

        # Attempt to export made intents

        if self.intents is not None:

            self.__export_intents__()

            self.console_log('Called __export_intents__() in __start__() [1]')

        # Attempt to import intents

        self.__import_intents__()

        self.console_log('Called __import_intents__() in __start__()')

        # Export intents again

        self.__export_intents__()

        self.console_log('Called __export_intents__() in __start__() [2]')

        # See if data pickle exists

        self.__import_data_pickle__()

        self.console_log('Called __import_data_pickle__() in __start__()')

        # Recreate tflearn model

        self.__create_tflearn_model__()

        self.console_log('Called __create_tflearn_model__() in __start__()')

        # Hard load tflearn model

        result = self.__load_tflearn_model__()

        self.console_log('Called __load_tflearn_model__() in __start__()')

        # Refit model if none exists

        if result is not True:

            self.__refit_tflearn_model__(progress=False, epochs=0)

            self.console_log('Called __refit_tflearn_model__() with lazy in __start__()')

        self.__load_tflearn_model__()

        self.console_log('Loading tflearn model [f]')

    def __fullstack_reload__(self):

        self.console_log('Starting __start__() as __fullstack_reload__()')

        self.__start__()

    def process(self, message, create_response=False):

        self.console_log(f'Processing new message "{message}"')

        results = self.model.predict([bag_of_words(message, self.words)])

        self.console_log('Model has predicted based on bag_of_words(*)')

        results_index = numpy.argmax(results)  # The index of the greatest value within the list

        self.console_log('Numpy has fetched most likely result')

        tag = self.labels[results_index]

        self.console_log(f'Assuming tag {tag}')

        intent = self.intents.get_intent_by_tag(tag)

        self.console_log(f'Assuming intent {intent}')

        self.console_log(f'Returning to be output[ tag: {tag}, intent: {intent} ]')

        if not create_response:

            return tag, intent

        else:

            return (tag, intent), random.choice(intent.responses)

    def reinforce(self, message, intent):

        self.console_log(f'Reinforcing "{message}" of {intent.tag}')

        intent = intent.__json__()
        intent['patterns'].append(message)
        self.__export_intents__(intent)

    def create_intent(self, tag: str, patterns: [str, ], responses: [str, ], context: str = ''):

        intent = {
            'tag': tag,
            'patterns': patterns,
            'responses': responses,
            'context': context,
            'emotion_modifiers': {
                'happiness': 0,
                'sadness': 0,
                'anger': 0,
                'fear': 0,
                'trust': 0,
                'disgust': 0,
                'anticipation': 0,
                'surprise': 0
            }
        }

        self.console_log(f'Creating new intent {intent}')

        self.__export_intents__(intent)

    def __init__(self, name):

        self.__bot_name__ = name

        self.intents = None
        self.words, self.labels, self.training, self.output = None, None, None, None

        self.net, self.model = None, None

        self.__training_process__ = 0
        self.__training_epochs__ = 0

        self.__console_log__ = []
        self.__console_log_dynos__ = []

        self.__start__()
        self.__load_tflearn_model__()

        self.console_log('Completed (own) __init__()')

    @property
    def name(self):

        return self.__bot_name__

    def console_log(self, message):

        entry = ConsoleEntry(message)

        self.__console_log__.append(entry)

        if log := Options.default_developer_console:

            log(self, entry)

deku = Bot('Deku')
gary1 = Bot('Gary1')
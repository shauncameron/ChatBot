import datetime


class Entry:

    def __repr__(self):

        return f'Entry @ {self.created}: {self.message}'

    def __init__(self, message):

        self.__message__ = message
        self.__created__ = datetime.datetime.now()
        self.read = False

    @property
    def message(self):

        return self.__message__

    @property
    def created(self):

        return self.__created__
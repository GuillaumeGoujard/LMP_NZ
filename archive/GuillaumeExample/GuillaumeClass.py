from lmpnz.GuillaumeExample import utilities

class guillaumeClass():
    def __init__(self, str):
        self.str = str

    def __repr__(self):
        return self.str

    def addition(self, a, b):
        return utilities.addition(a,b)
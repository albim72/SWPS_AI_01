from random import random


def ms1(history_self, history_opponent):
    if not history_opponent:
        return "C"
    if history_opponent[-1] == "D":
        return "D"
    return "C"

def ms2(history_self, history_opponent):
    return random.choice(["C", "D"])

def ms3(history_self, history_opponent):
    if len(history_self) % 3 == 2:
        return "D"
    return "C"

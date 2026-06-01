# test_wrapper.py
from predict_wrapper import predict_from_dict

sample = {
    "driver": "hamilton",
    "constructor": "mercedes",
    "race": "monaco grand prix",
    "grid": 1,
    "laps": 78,
    "points": 25
}

if __name__ == "__main__":
    print(predict_from_dict(sample))

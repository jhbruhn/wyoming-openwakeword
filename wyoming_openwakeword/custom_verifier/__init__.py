from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
import logging
import collections
import numpy as np
import glob
import pathlib
import pickle
import hashlib
import pathlib

_LOGGER = logging.getLogger()

POSITIVE_DIR_NAME = "positive"
NEGATIVE_DIR_NAME = "negative"


@dataclass
class VerificationResult:
    probability: float
    speaker: Optional[str] = None


class CustomVerifier(ABC):
    @abstractmethod
    def __init__(
        self, positive_samples: dict, negative_samples: list, model_path: str, model_name: str
    ):
        pass

    @abstractmethod
    def predict(self, features) -> VerificationResult:
        pass


from .logistic_regression import *


class CustomVerifierManager:
    def __init__(self, sample_dir: str):
        self.sample_dir = sample_dir
        self.verifier_cache = {}

    def get_verifier(self, model_path: str, model_name: str, type: str = "LogisticRegression") -> CustomVerifier:
        id = str(model_path) + "-" + model_name

        if id in self.verifier_cache.keys():
            return self.verifier_cache[id]
        
        positive_dir = f"{self.sample_dir}/{model_name}/{POSITIVE_DIR_NAME}/"
        
        negative_dir = f"{self.sample_dir}/{model_name}/{NEGATIVE_DIR_NAME}/"
        
        # list all subdirectories (due to trailing slash) -> speaker names
        speaker_names = list(map(lambda n: pathlib.PurePath(n).name, glob.glob(f"{positive_dir}/*/")))

        positive_samples = {}
        for name in speaker_names:
            positive_samples[name] = glob.glob(f"{positive_dir}/{name}/*.wav")
        
        negative_samples = glob.glob(f"{negative_dir}/*.wav")

        # calculate digest for caching
        digest = hashlib.md5()
        file_buffer  = bytearray(128*1024)
        file_view = memoryview(file_buffer)
        speaker_names.sort()
        for name in speaker_names:
            digest.update(name.encode('utf-8'))
            positive_samples[name].sort()
            for file in positive_samples[name]:
                with open(file, 'rb', buffering=0) as file:
                    for n in iter(lambda : file.readinto(file_view), 0):
                        digest.update(file_view[:n])
        negative_samples.sort()
        for file in negative_samples:
            with open(file, 'rb', buffering=0) as file:
                for n in iter(lambda : file.readinto(file_view), 0):
                    digest.update(file_view[:n])
        verifier_hash = digest.hexdigest()

        # try to load cached pickle
        pickle_path = f"{self.sample_dir}/{model_name}/{type}-{verifier_hash}.pkl"
        verifier = None
        try:
            verifier = pickle.load(open(pickle_path, 'rb'))
            _LOGGER.debug(f"Found {type} verifier for {model_name} at {pickle_path}")
        except:
            _LOGGER.debug(f"No {type} verifier for {model_name} cached, training new one.")
        
            # currently only one type of classifier exists
            verifier = LogisticRegressionCustomVerifier(positive_samples, negative_samples, str(model_path), model_name)
            _LOGGER.debug(f"Saving {type} verifier for {model_name} to {pickle_path}")
            pickle.dump(verifier, open(pickle_path, "wb"))
        
        self.verifier_cache[id] = verifier

        return verifier


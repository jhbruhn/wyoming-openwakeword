from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
import logging
import collections
import numpy as np
import glob
import scipy
import openwakeword
import pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import CustomVerifier, VerificationResult

_LOGGER = logging.getLogger()

def flatten_features(x):
    return [i.flatten() for i in x]

class LogisticRegressionCustomVerifier(CustomVerifier):
    classifier: Pipeline

    def __init__(
        self, positive_samples: dict, negative_samples: list, model_path: str, model_name: str
    ):
        model = openwakeword.Model(
            wakeword_models=[model_path], inference_framework="tflite",
            melspec_model_path=str(pathlib.Path(__file__).parent.resolve()) + "/../models/melspectrogram.tflite",
            embedding_model_path=str(pathlib.Path(__file__).parent.resolve()) +"/../models/embedding_model.tflite"
        )

        speaker_names = list(positive_samples.keys())

        labels = []
        positive_features = None

        # Get features from positive reference clips
        for speaker_name in speaker_names:
            _LOGGER.info(f"Processing positive reference clips for '{speaker_name}'")
            positive_features_speaker = np.vstack(
                [
                    _get_reference_clip_features(i, model, model_name, N=5)
                    for i in positive_samples[speaker_name]
                ]
            )
            if positive_features_speaker.shape[0] == 0:
                raise ValueError(
                    "The positive features were not created! Make sure that"
                    " the positive reference clips contain the appropriate audio"
                    " for the desired model."
                )

            if positive_features is None:
                positive_features = positive_features_speaker
            else:
                positive_features = np.vstack(
                    (positive_features, positive_features_speaker)
                )

            labels += [speaker_name] * positive_features_speaker.shape[0]

        _LOGGER.info("Processing negative reference clips")
        negative_features = np.vstack(
            [
                _get_reference_clip_features(i, model, model_name, threshold=0.0, N=1)
                for i in negative_samples
            ]
        )
        labels += [""] * negative_features.shape[0]

        _LOGGER.info("Training and saving verifier model...")
        features = np.vstack((positive_features, negative_features))

        clf = LogisticRegression(random_state=0, max_iter=2000, C=0.001)
        pipeline = make_pipeline(FunctionTransformer(flatten_features), StandardScaler(), clf)
        pipeline.fit(features, labels)
        
        self.classifier = pipeline


    def predict(self, features) -> VerificationResult:
        NEGATIVE_CLASS_NAME = ""

        probabilities = self.classifier.predict_proba(features)[0]
        
        negative_class_index = self.classifier.classes_.tolist().index(NEGATIVE_CLASS_NAME)
        negative_probability = probabilities[negative_class_index]
        
        max_probability_index = np.argmax(probabilities)
        
        speaker = self.classifier.classes_[max_probability_index]
        if speaker == NEGATIVE_CLASS_NAME:
            speaker = None
        
        return VerificationResult(probability=1.0 - negative_probability, speaker=speaker)


def _get_reference_clip_features(
    reference_clip: str,
    oww_model: openwakeword.Model,
    model_name: str,
    threshold: float = 0.5,
    N: int = 3,
):
    positive_data = collections.defaultdict(list)

    for _ in range(N):
        # Load clip
        if type(reference_clip) == str:
            sr, dat = scipy.io.wavfile.read(reference_clip)
        else:
            dat = reference_clip

        # Set random starting point to get small variations in features
        if N != 1:
            dat = dat[np.random.randint(0, 1280) :]

        # Get predictions
        step_size = 1280
        for i in range(0, dat.shape[0] - step_size, step_size):
            predictions = oww_model.predict(dat[i : i + step_size])
            model_name = list(predictions.keys())[0] # taking the first one for now
            if predictions[model_name] >= threshold:
                features = oww_model.preprocessor.get_features(  # type: ignore[has-type]
                    oww_model.model_inputs[model_name]  # type: ignore[has-type]
                )
                positive_data[model_name].append(features)

    if len(positive_data[model_name]) == 0:
        positive_data[model_name].append(
            np.empty((0, oww_model.model_inputs[model_name], 96))
        )  # type: ignore[has-type]

    return np.vstack(positive_data[model_name])


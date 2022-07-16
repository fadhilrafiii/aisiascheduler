import logging

from services.ml import retrain_model

logger = logging.getLogger(__name__)


def retrain_model_job():
    retrain_model()

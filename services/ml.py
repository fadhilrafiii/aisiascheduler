import logging

logger = logging.getLogger(__name__)


def retrain_model():
    import time

    logger.info("I'm training your model now!")
    time.sleep(5)

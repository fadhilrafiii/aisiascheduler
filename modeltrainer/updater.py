import logging

from apscheduler.schedulers.background import BackgroundScheduler
from django.conf import settings

from modeltrainer.job import retrain_model_job

logger = logging.getLogger("django")


def start():
    MINUTE, HOUR, DAY = ('minute', 'hour', 'day')
    SCHEDULER_INTERVAL_NUMBER = getattr(settings, "SCHEDULER_INTERVAL_NUMBER", None)
    SCHEDULER_INTERVAL_UNIT = getattr(settings, "SCHEDULER_INTERVAL_UNIT", None)

    if SCHEDULER_INTERVAL_NUMBER and SCHEDULER_INTERVAL_UNIT:
        logger.info(
            f"Scheduler is starting! Retraining will be triggered every {SCHEDULER_INTERVAL_NUMBER} {SCHEDULER_INTERVAL_UNIT}s"
        )
        scheduler = BackgroundScheduler()

        if (SCHEDULER_INTERVAL_UNIT == MINUTE):
            scheduler.add_job(retrain_model_job, "interval", minutes=int(SCHEDULER_INTERVAL_NUMBER))
        elif (SCHEDULER_INTERVAL_UNIT == HOUR):
            scheduler.add_job(retrain_model_job, "interval", hours=int(SCHEDULER_INTERVAL_NUMBER))
        elif (SCHEDULER_INTERVAL_UNIT == DAY):
            scheduler.add_job(retrain_model_job, "interval", day=int(SCHEDULER_INTERVAL_NUMBER))

        scheduler.start()
    else:
        logger.error("Scheduler interval is not defined!")

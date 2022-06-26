from django.conf import settings
from apscheduler.schedulers.background import BackgroundScheduler
from modeltrainer.job import retrain_model
import logging

logger = logging.getLogger('django')

def start():
  SCHEDULER_INTERVAL=getattr(settings, "SCHEDULER_INTERVAL", None)

  if (SCHEDULER_INTERVAL):
    logger.info(f'Scheduler is starting!\nRetraining will be triggered every {SCHEDULER_INTERVAL}' )
    scheduler = BackgroundScheduler()
    scheduler.add_job(retrain_model, 'interval', minutes=int(SCHEDULER_INTERVAL))
    scheduler.start()
  else:
    logger.error('Scheduler interval is not defined!')

  
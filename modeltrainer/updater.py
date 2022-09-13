import logging

from apscheduler.schedulers.background import BackgroundScheduler
from modeltrainer.constants import FASTERRCNN, MOBILE_NET

from modeltrainer.job import retrain_model_job

logger = logging.getLogger("django")


def start_mobile_net_scheduler():
    logger.info("Scheduler is starting!")
    scheduler = BackgroundScheduler(timezone='Asia/Jakarta')
    
    logger.info("MobileNet retraining will triggered every MONDAY on 00.00!")
    scheduler.add_job(retrain_model_job, 'cron', day_of_week='wed', hour= '2', minute='47', args= [MOBILE_NET])
    
    logger.info("FasterRCNN retraining will triggered every FRIDAY on 00.00!")
    scheduler.add_job(retrain_model_job, 'cron', day_of_week='fri', hour= '0', args= [FASTERRCNN])

    scheduler.start()
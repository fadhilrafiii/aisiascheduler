import logging

from django.utils.timezone import now

from modeltrainer.constants import FAILED
from modeltrainer.constants import ON_PROGRESS
from modeltrainer.constants import SUCCESS
from modeltrainer.models import AisiaState
from modeltrainer.utils import create_new_retrain_state
from modeltrainer.utils import update_retrain_state
from services.ml import retrain_model

logger = logging.getLogger(__name__)


def retrain_model_job():
    start_time = now()
    try:
        previous_retrain_status = AisiaState.objects.last()
        logger.info(f'Last retrain status: {vars(previous_retrain_status) if (previous_retrain_status) else None}')

        if (previous_retrain_status is None or previous_retrain_status.status != ON_PROGRESS):
            curr_state = create_new_retrain_state()
            start_time = curr_state.date_started
            logger.info(f'Retrain start at: {start_time}')

            # Retrain Model
            retrain_model()
            # Update state
            update_retrain_state(status=SUCCESS, start_time=start_time)
            logger.info(f'Retrain {SUCCESS} at {now()}')
        else:
            logger.info('Job retrain model is not started due to last retrain is not finished yet!')
    except Exception as e:
        logger.error(f'An error occured: {str(e)}')
        logger.warn(f'Retrain {SUCCESS} at {now()}')
        # Update state
        update_retrain_state(status=FAILED, start_time=start_time)

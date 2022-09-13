import logging

from django.utils.timezone import now
from rest_framework import serializers

from modeltrainer.constants import ON_PROGRESS
from modeltrainer.models import AisiaState
from modeltrainer.serializers import AisiaStateSerializer


logger = logging.getLogger(__name__)


def create_new_retrain_state(model_name):
    try:
        created_state = AisiaState.objects.create(
            model_name=model_name,
            status=ON_PROGRESS
        )

        return created_state
    except serializers.ValidationError as e:
        logger.error(f'An error occured: {str(e)}')
    except Exception as exception:
        raise exception


def update_retrain_state(state_id, status, start_time):
    current_retrain_status = AisiaState.objects.get(id=state_id)
    current_retrain_status.status = status

    finish_time = now()
    current_retrain_status.updated_date = finish_time

    time_needed = (finish_time - start_time).total_seconds()
    current_retrain_status.time_needed = int(time_needed)  # in seconds (integer)
   
    model_accuracy = get_accuracy_from_file()
    current_retrain_status.model_accuracy = model_accuracy
    print('INSIDE', current_retrain_status.id)
    current_retrain_status.save()
    
def get_accuracy_from_file(file_path='logs/accuracy.log'):
    f = open(file_path, 'r')
    
    data = f.read()
    lines = data.split('\n')
    target_mAP_line = lines[6]
    
    target_line_splitted_by_equal_sign = target_mAP_line.split(' = ')
    mAP = float(target_line_splitted_by_equal_sign.pop())
    
    return mAP
        

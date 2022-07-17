import logging

from django.utils.timezone import now
from rest_framework import serializers

from modeltrainer.constants import ON_PROGRESS
from modeltrainer.models import AisiaState
from modeltrainer.serializers import AisiaStateSerializer


logger = logging.getLogger(__name__)


def create_new_retrain_state():
    try:
        state = {
            "total_new_input": 0,
            "status": ON_PROGRESS
        }
        validated_state = AisiaStateSerializer(data=state)

        if (not validated_state.is_valid()):
            raise serializers.ValidationError(validated_state.errors)

        validated_state.save()

        created_state = AisiaState.objects.last()
        return created_state
    except serializers.ValidationError as e:
        logger.error(f'An error occured: {str(e)}')
    except Exception as exception:
        raise exception


def update_retrain_state(status, start_time):
    current_retrain_status = AisiaState.objects.last()
    current_retrain_status.status = status

    finish_time = now()
    current_retrain_status.date_finished = finish_time

    time_needed = (finish_time - start_time).total_seconds()
    current_retrain_status.time_needed = int(time_needed)  # in seconds (integer)
    current_retrain_status.save()

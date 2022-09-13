from django.utils import timezone
from djongo import models

from modeltrainer.constants import FAILED, MODEL_NAME
from modeltrainer.constants import STATUS_OPTIONS


class AisiaState(models.Model):
    status = models.CharField(default=FAILED, max_length=12, choices=STATUS_OPTIONS, blank=True, null=True)
    model_name = models.CharField(max_length=16, choices=MODEL_NAME, blank=False, null=False)
    created_date = models.DateTimeField(auto_now_add=timezone.now)
    updated_date = models.DateTimeField(blank=True, null=True)
    time_needed = models.IntegerField(default=0, blank=True, null=True)
    model_accuracy = models.FloatField(default=0.0, blank=True, null=True)

    class Meta:
        db_table = "aisia_state"

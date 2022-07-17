from django.utils import timezone
from djongo import models

from modeltrainer.constants import FAILED
from modeltrainer.constants import STATUS_OPTIONS

# Create your models here.


class AisiaState(models.Model):
    date_started = models.DateTimeField(auto_now_add=timezone.now)
    date_finished = models.DateTimeField(blank=True, null=True)
    total_new_input = models.IntegerField(default=0, blank=True, null=True)
    status = models.CharField(default=FAILED, max_length=12, choices=STATUS_OPTIONS, blank=True, null=True)
    time_needed = models.IntegerField(default=0, blank=True, null=True)

    class Meta:
        db_table = "aisia_state"

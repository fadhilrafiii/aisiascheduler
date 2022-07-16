from django.utils import timezone
from djongo import models

# Create your models here.


class AisiaState(models.Model):
    ON_PROGRESS, SUCCESS, FAILED = ("ON_PROGRESS", "SUCCESS", "FAILED")

    timestamp = models.DateTimeField(auto_now=timezone.now)
    total_new_input = models.IntegerField(default=0)
    status = models.CharField(default="FAILED", max_length=6)
    time_needed = models.IntegerField(default=0)

    class Meta:
        db_table = "aisia_state"

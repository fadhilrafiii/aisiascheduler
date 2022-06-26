from djongo import models
from django.utils import timezone

# Create your models here.
class AisiaState(models.Model):
    timestamp = models.DateTimeField(auto_now=timezone.now)
    total_new_input = models.IntegerField(default=0)
    last_train_status = models.CharField(default="FAILED", max_length=6)
    time_needed = models.IntegerField(default=0)

    class Meta:
        db_table = "aisia_state"

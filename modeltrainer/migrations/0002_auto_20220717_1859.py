# Generated by Django 3.2.14 on 2022-07-17 11:59
import django.utils.timezone
from django.db import migrations
from django.db import models


class Migration(migrations.Migration):

    dependencies = [
        ('modeltrainer', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='aisiastate',
            name='last_train_status',
        ),
        migrations.RemoveField(
            model_name='aisiastate',
            name='timestamp',
        ),
        migrations.AddField(
            model_name='aisiastate',
            name='date_finished',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='aisiastate',
            name='date_started',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='aisiastate',
            name='status',
            field=models.CharField(blank=True, choices=[('ON_PROGRESS', 'ON_PROGRESS'), ('SUCCESS', 'SUCCESS'), (
                'FAILED', 'FAILED'), ('UNSTARTED', 'UNSTARTED')], default='FAILED', max_length=12, null=True),
        ),
        migrations.AlterField(
            model_name='aisiastate',
            name='time_needed',
            field=models.IntegerField(blank=True, default=0, null=True),
        ),
        migrations.AlterField(
            model_name='aisiastate',
            name='total_new_input',
            field=models.IntegerField(blank=True, default=0, null=True),
        ),
    ]

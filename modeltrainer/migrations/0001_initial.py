# Generated by Django 3.2.13 on 2022-06-26 10:29
from django.db import migrations
from django.db import models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="AisiaState",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("timestamp", models.DateTimeField(auto_now=True)),
                ("total_new_input", models.IntegerField(default=0)),
                ("last_train_status", models.CharField(default="FAILED", max_length=6)),
                ("time_needed", models.IntegerField(default=0)),
            ],
            options={
                "db_table": "aisia_state",
            },
        ),
    ]

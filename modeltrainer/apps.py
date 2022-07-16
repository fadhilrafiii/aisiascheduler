from django.apps import AppConfig


class ModeltrainerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "modeltrainer"

    def ready(self):
        from modeltrainer.updater import start

        # Start the scheduler
        start()

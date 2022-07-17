from rest_framework import serializers

from .models import AisiaState


class AisiaStateSerializer(serializers.ModelSerializer):
    class Meta:
        model = AisiaState
        fields = "__all__"

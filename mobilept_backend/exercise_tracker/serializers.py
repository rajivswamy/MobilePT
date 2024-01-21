from rest_framework import serializers
from .models import ReferenceSession, ExerciseSession, ExerciseROMData, ReferenceROMData

class ReferenceSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReferenceSession
        fields = '__all__'

class ExerciseSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExerciseSession
        fields = '__all__'

class ExerciseROMDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExerciseROMData
        fields = '__all__'

class ReferenceROMDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReferenceROMData
        fields = '__all__'
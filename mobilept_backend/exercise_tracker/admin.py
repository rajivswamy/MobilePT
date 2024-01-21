from django.contrib import admin
from exercise_tracker.models import ReferenceSession, ExerciseSession, ExerciseROMData, ReferenceROMData

# Register your models here.
admin.site.register(ReferenceSession)
admin.site.register(ExerciseSession)
admin.site.register(ReferenceROMData)
admin.site.register(ExerciseROMData)



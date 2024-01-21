from django.urls import path
from exercise_tracker import views

urlpatterns = [
    path("", views.home, name="home"),
    path("reference_sessions/", views.References.as_view()),
    path("reference_sessions/<str:pk>/", views.ReferenceDetail.as_view()),
    path("reference_rom_data/", views.ReferenceROM.as_view()),
    path("reference_rom_data/<str:pk>", views.ReferenceROMDetail.as_view()),
    path("exercise_sessions/", views.ExerciseSessions.as_view()),
    path("exercise_sessions/<str:pk>/", views.ExerciseSessionsDetail.as_view()),
    path("exercise_rom_data/", views.ExerciseROM.as_view()),
    path("exercise_rom_data/<str:pk>", views.ExerciseROMDetail.as_view()),
    path("dtw/",views.DTW.as_view()),
]
from django.db import models
from django.core.serializers.json import DjangoJSONEncoder
import uuid

EXERCISE_CHOICES = [('SQUAT', 'SQUAT'),
               ('ARM_RAISE','ARM RAISE'),
               ('LEG_RAISE', 'LEG RAISE'),
               ('BICEP_CURL','BICEP CURL ALTERNATING'),]

JOINT_CHOICES = [('L_KNEE', 'LEFT KNEE'), 
                ('R_KNEE', 'RIGHT KNEE'),
                ('L_HIP', 'LEFT HIP'),
                ('R_HIP', 'RIGHT HIP'),
                ('L_ELBOW', 'LEFT ELBOW'),
                ('R_ELBOW', 'RIGHT ELBOw'),
                ('L_SHOULDER', 'LEFT SHOULDER'),
                ('R_SHOULDER', 'RIGHT_SHOULDER')]

EXERCISE_JOINTS = {'SQUAT': ['L_KNEE', 'R_KNEE', 'L_HIP','R_HIP', 'L_SHOULDER','R_SHOULDER'],
                     'ARM_RAISE': ['L_SHOULDER','R_SHOULDER'],
                     'LEG_RAISE': ['L_HIP','R_HIP'],
                     'BICEP_CURL':['L_ELBOW','R_ELBOW']}

ALL_JOINTS = ['L_KNEE', 'R_KNEE', 'L_HIP', 'R_HIP', 'L_ELBOW', 'R_ELBOW', 
              'L_SHOULDER', 'R_SHOULDER']


# ROM Movements for each exercise and joints of interest:
# Squat: Knees - H to L, Hips: H to L
# Arm Raise - Shoulders: L to H
# Leg Raise - Hips: H to L
# Bicep Curl - Elbow: H to L

# Store a reference pose of a user under the supervision of a clinician
class ReferenceSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Exercise metadata
    name = models.CharField(max_length=60, unique=True) # User set name for display
    exercise = models.CharField(max_length=20, choices=EXERCISE_CHOICES)
    target_reps = models.PositiveIntegerField()
    detected_reps = models.PositiveIntegerField()
    date = models.DateTimeField()

    # Store pose estimation keypoints for current pose
    pose_keypoints = models.JSONField(null=True)

    # Store pose estimation keypoints for arbitrary sequence of frames
    keypoints_sequence = models.JSONField()

    # Analytics from model: ROM data, feedback
    analytics = models.JSONField(null=True)

# Store information regarding a user's exercise session 
class ExerciseSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Exercise metadata
    name = models.CharField(max_length=60, unique=True) # User set name for display
    exercise = models.CharField(max_length=40, choices=EXERCISE_CHOICES)
    target_reps = models.PositiveIntegerField()
    detected_reps = models.PositiveIntegerField()
    sets = models.PositiveIntegerField()
    date = models.DateTimeField()

    # Corresponding reference
    reference = models.ForeignKey(ReferenceSession, on_delete=models.SET_NULL, null=True)

    # Store pose estimation keypoints for current pose
    pose_keypoints = models.JSONField(null=True)

    # Store pose estimation keypoints for arbitrary sequence of frames
    keypoints_sequence = models.JSONField()

    # Analytics from model: ROM data, feedback
    analytics = models.JSONField(null=True)

    # Accuracy Score: Based on Reference exercise, will be null if reference is null
    accuracy = models.FloatField(null=True)

class ExerciseROMData(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    joint = models.CharField(max_length=40, choices=JOINT_CHOICES)
    date = models.DateTimeField()
    exercise = models.CharField(max_length=40, choices=EXERCISE_CHOICES)
   
    # Data points
    data = models.JSONField()
    minimum = models.FloatField()
    maximum = models.FloatField()

    exercise_session = models.ForeignKey(ExerciseSession, null=True,  on_delete=models.CASCADE)

class ReferenceROMData(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    joint = models.CharField(max_length=40, choices=JOINT_CHOICES)
    date = models.DateTimeField()
    exercise = models.CharField(max_length=40, choices=EXERCISE_CHOICES)
   
    # Data points
    data = models.JSONField()
    minimum = models.FloatField()
    maximum = models.FloatField()


    reference_session = models.ForeignKey(ReferenceSession, null=True,  on_delete=models.CASCADE)







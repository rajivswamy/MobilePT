# Generated by Django 4.1.3 on 2023-01-04 05:33

from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ReferenceSession",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("name", models.CharField(max_length=60, unique=True)),
                (
                    "exercise",
                    models.CharField(
                        choices=[
                            ("SQUAT", "SQUAT"),
                            ("ARM_RAISE", "ARM RAISE"),
                            ("LEG_RAISE", "LEG RAISE"),
                            ("BICEP_CURL", "BICEP CURL ALTERNATING"),
                        ],
                        max_length=20,
                    ),
                ),
                ("target_reps", models.PositiveIntegerField()),
                ("detected_reps", models.PositiveIntegerField()),
                ("date", models.DateTimeField()),
                ("pose_keypoints", models.JSONField(null=True)),
                ("keypoints_sequence", models.JSONField()),
                ("analytics", models.JSONField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name="ReferenceROMData",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "joint",
                    models.CharField(
                        choices=[
                            ("L_KNEE", "LEFT KNEE"),
                            ("R_KNEE", "RIGHT KNEE"),
                            ("L_HIP", "LEFT HIP"),
                            ("R_HIP", "RIGHT HIP"),
                            ("L_ELBOW", "LEFT ELBOW"),
                            ("R_ELBOW", "RIGHT ELBOw"),
                            ("L_SHOULDER", "LEFT SHOULDER"),
                            ("R_SHOULDER", "RIGHT_SHOULDER"),
                        ],
                        max_length=40,
                    ),
                ),
                ("date", models.DateTimeField()),
                (
                    "exercise",
                    models.CharField(
                        choices=[
                            ("SQUAT", "SQUAT"),
                            ("ARM_RAISE", "ARM RAISE"),
                            ("LEG_RAISE", "LEG RAISE"),
                            ("BICEP_CURL", "BICEP CURL ALTERNATING"),
                        ],
                        max_length=40,
                    ),
                ),
                ("data", models.JSONField()),
                ("minimum", models.FloatField()),
                ("maximum", models.FloatField()),
                (
                    "reference_session",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        to="exercise_tracker.referencesession",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="ExerciseSession",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("name", models.CharField(max_length=60, unique=True)),
                (
                    "exercise",
                    models.CharField(
                        choices=[
                            ("SQUAT", "SQUAT"),
                            ("ARM_RAISE", "ARM RAISE"),
                            ("LEG_RAISE", "LEG RAISE"),
                            ("BICEP_CURL", "BICEP CURL ALTERNATING"),
                        ],
                        max_length=40,
                    ),
                ),
                ("target_reps", models.PositiveIntegerField()),
                ("detected_reps", models.PositiveIntegerField()),
                ("sets", models.PositiveIntegerField()),
                ("date", models.DateTimeField()),
                ("pose_keypoints", models.JSONField(null=True)),
                ("keypoints_sequence", models.JSONField()),
                ("analytics", models.JSONField(null=True)),
                ("accuracy", models.FloatField(null=True)),
                (
                    "reference",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        to="exercise_tracker.referencesession",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="ExerciseROMData",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "joint",
                    models.CharField(
                        choices=[
                            ("L_KNEE", "LEFT KNEE"),
                            ("R_KNEE", "RIGHT KNEE"),
                            ("L_HIP", "LEFT HIP"),
                            ("R_HIP", "RIGHT HIP"),
                            ("L_ELBOW", "LEFT ELBOW"),
                            ("R_ELBOW", "RIGHT ELBOw"),
                            ("L_SHOULDER", "LEFT SHOULDER"),
                            ("R_SHOULDER", "RIGHT_SHOULDER"),
                        ],
                        max_length=40,
                    ),
                ),
                ("date", models.DateTimeField()),
                (
                    "exercise",
                    models.CharField(
                        choices=[
                            ("SQUAT", "SQUAT"),
                            ("ARM_RAISE", "ARM RAISE"),
                            ("LEG_RAISE", "LEG RAISE"),
                            ("BICEP_CURL", "BICEP CURL ALTERNATING"),
                        ],
                        max_length=40,
                    ),
                ),
                ("data", models.JSONField()),
                ("minimum", models.FloatField()),
                ("maximum", models.FloatField()),
                (
                    "exercise_session",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        to="exercise_tracker.exercisesession",
                    ),
                ),
            ],
        ),
    ]
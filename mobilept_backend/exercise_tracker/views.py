from .models import ReferenceSession, ExerciseSession, ExerciseROMData, ReferenceROMData, EXERCISE_JOINTS, ALL_JOINTS
from .serializers import ReferenceSessionSerializer, ExerciseSessionSerializer, ExerciseROMDataSerializer, ReferenceROMDataSerializer


from django.http import HttpResponse
from django.http import Http404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from datetime import datetime
import sys
import uuid

# Rep counter
from .utils import rep_counter, rom_processing


def home(request):
    return HttpResponse("Hello, Django!")


class References(APIView):
    """
    List all references data or create a new reference
    """
    def get(self, request):
        references = ReferenceSession.objects.all()
        serializer = ReferenceSessionSerializer(references, many=True)
        return Response(serializer.data)
    

    # Largest chunk of work needed
    def post(self, request):

        # Get data from request
        payload = request.data

        # Validate data
        flag = self.validate_reference_payload(payload)

        if flag is False:
            return Response(status=status.HTTP_400_BAD_REQUEST)

        # Count reps from session data
        # Input: pose keypoints of video, time array, 
        # Output: number of detected reps, array of rep count over frames       

        flag, output = rep_counter.calculate_reps(payload['keypoints_sequence']['pose_landmarks'],
                                                  payload['keypoints_sequence']['frame_width'],
                                                  payload['keypoints_sequence']['frame_height'],
                                                  payload['exercise'])
        if flag is False:
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Add data to payload
        payload['detected_reps'] = output['detected_reps']
        payload['keypoints_sequence']['reps_frame_array'] = output['reps_frame_array']

        # Get ROM Data, joints depends on exercise
        # Input: world key points, joints of interest
        # Output: smoothed and raw signals for joints of interest
        # each exercise will have list of joints of interest
        joints_of_int = EXERCISE_JOINTS[payload['exercise']]
        pose_kps = payload['keypoints_sequence']['pose_world_landmarks']

        flag, output = rom_processing.process_joint_signals(pose_kps, ALL_JOINTS)
        if flag is False:
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Store ROM Analytics in analytics field of object
        payload['analytics'] = {'ROM':output}
        # Set pk for new reference session
        payload['id'] = uuid.uuid4()

        # Validate with serializer and save object to database
        serializer = ReferenceSessionSerializer(data=payload)
        if serializer.is_valid():
            ref_obj = serializer.save()
        
        # Create Reference ROM objects using analytics
        for joint in joints_of_int:

            data = payload['analytics']['ROM'][joint]

            # Prep rom data payload for object
            rom_payload = {
                'id': uuid.uuid4(),
                'joint': joint,
                'exercise': ref_obj.id,
                'reference_session': ref_obj,
                'date': ref_obj.date,
                'data': data,
                'minimum': data['avg_min'],
                'maximum': data['avg_max']
            }

            # Create and save object
            rom = ReferenceROMData(**rom_payload)
            rom.save()
            


        # Return response        
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    def delete(self, request):
        ReferenceSession.objects.all().delete()
        return Response(status=status.HTTP_200_OK)
    
    def validate_reference_payload(self, payload):
        try:
            choices = ['SQUAT','ARM_RAISE','LEG_RAISE','BICEP_CURL']
            
            if payload['exercise'] not in choices:
                return False

        except Exception as ex:
            print(ex, file=sys.stderr)
            return False

        return True

class ReferenceDetail(APIView):
    """
    Retreive, update, or delete a reference instance
    """
    def get_object(self, pk):
        try:
            return ReferenceSession.objects.get(pk=pk)
        except ReferenceSession.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        reference = self.get_object(pk)
        serializer = ReferenceSessionSerializer(reference)
        return Response(serializer.data)
    
    def delete(self, request, pk, format=None):
        reference = self.get_object(pk)
        reference.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ReferenceROM(APIView):
    def get(self, request):

        roms = ReferenceROMData.objects.all()
        
        # filter by exercise name
        exercise = request.query_params.get('exercise')
        if exercise is not None:
            roms = roms.filter(exercise=exercise)

        # filter by start and end date
        start_date_str = request.query_params.get('start_date')
        if start_date_str is not None:
            start = datetime.fromisoformat(start_date_str)
            roms = roms.filter(date__gte=start)

        # filter by start and end date
        end_date_str = request.query_params.get('end_date')
        if end_date_str is not None:
            end = datetime.fromisoformat(end_date_str)
            roms = roms.filter(date__lte=end)

        joint = request.query_params.get('joint')
        if joint is not None:
            roms = roms.filter(joint=joint)
        
        reference_id = request.query_params.get('reference_id')
        if reference_id is not None:
            roms = roms.filter(reference_session__pk=reference_id)

        serializer = ReferenceROMDataSerializer(roms, many=True)
        return Response(serializer.data)



class ReferenceROMDetail(APIView):
    """
    Retreive, update, or delete a reference instance
    """
    def get_object(self, pk):
        try:
            return ReferenceROMData.objects.get(pk=pk)
        except ReferenceROMDataSerializer.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        reference = self.get_object(pk)
        serializer = ReferenceROMDataSerializer(reference)
        return Response(serializer.data)

class ExerciseSessions(APIView):
    """
    List all references data or create a new reference
    """
    def get(self, request):
        
        exercises = ExerciseSession.objects.all()

        reference_id = request.query_params.get('reference_id')
        if reference_id is not None:
            exercises = exercises.filter(reference__pk=reference_id)

        serializer = ExerciseSessionSerializer(exercises, many=True)
        return Response(serializer.data)
    

    # Add new exercise session to the database
    def post(self, request):

        # Get data from request
        payload = request.data

        flag = self.validate_exercise_payload(payload)

        if flag is False:
            return Response(status=status.HTTP_400_BAD_REQUEST)

        # Count reps from session data
        # Input: pose keypoints of video, time array, 
        # Output: number of detected reps, array of rep count over frames       

        flag, output = rep_counter.calculate_reps(payload['keypoints_sequence']['pose_landmarks'],
                                                  payload['keypoints_sequence']['frame_width'],
                                                  payload['keypoints_sequence']['frame_height'],
                                                  payload['exercise'])
        if flag is False:
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Add data to payload
        payload['detected_reps'] = output['detected_reps']
        payload['keypoints_sequence']['reps_frame_array'] = output['reps_frame_array']

        # Get ROM Data, joints depends on exercise
        # Input: world key points, joints of interest
        # Output: smoothed and raw signals for joints of interest
        # each exercise will have list of joints of interest
        joints_of_int = EXERCISE_JOINTS[payload['exercise']]
        pose_kps = payload['keypoints_sequence']['pose_world_landmarks']

        flag, output = rom_processing.process_joint_signals(pose_kps, ALL_JOINTS)
        if flag is False:
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Store ROM Analytics in analytics field of object
        payload['analytics'] = {'ROM':output}
        # Set pk for new reference session
        payload['id'] = uuid.uuid4()

        # placeholder value for accuracy score
        payload['accuracy'] = None


        # Execute DTW code if the reference id is given in the payload
        if 'reference' in payload:
            ref = ReferenceSession.objects.get(pk=payload['reference'])
            flag, data  = rom_processing.dtw_exercise(payload, ref)
            if flag is False:
                return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            payload['analytics']['DTW'] = data

            payload['accuracy'] = data['joint_angles_indiv_joi']['accuracy']

        # Validate with serializer and save object to database
        serializer = ExerciseSessionSerializer(data=payload)
        if serializer.is_valid():
            ex_obj = serializer.save()
        else:
            print(serializer.errors)
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
        # Create Reference ROM objects using analytics
        for joint in joints_of_int:

            data = payload['analytics']['ROM'][joint]
            # Prep rom data payload for object
            rom_payload = {
                'id': uuid.uuid4(),
                'joint': joint,
                'exercise': ex_obj.exercise,
                'exercise_session': ex_obj,
                'date': ex_obj.date,
                'data': data,
                'minimum': data['avg_min'],
                'maximum': data['avg_max']
            }

            # Create and save object
            rom = ExerciseROMData(**rom_payload)
            rom.save()


        return Response(serializer.data, status.HTTP_201_CREATED)
    
    def delete(self, request):
        ExerciseSession.objects.all().delete()
        return Response(status=status.HTTP_200_OK)


    def validate_exercise_payload(self, payload):
        try:
            choices = ['SQUAT','ARM_RAISE','LEG_RAISE','BICEP_CURL']
            
            if payload['exercise'] not in choices:
                return False

            # if the request specifies a reference, then check the ref
            if 'reference' in payload:
                ref = ReferenceSession.objects.get(pk=payload['reference'])

                if ref.exercise != payload['exercise']:
                    return False

        except Exception as ex:
            print(ex, file=sys.stderr)
            return False

        return True
    
    

class ExerciseSessionsDetail(APIView):
    """
    Retreive, update, or delete a reference instance
    """
    def get_object(self, pk):
        try:
            return ExerciseSession.objects.get(pk=pk)
        except ExerciseSession.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        exercise = self.get_object(pk)
        serializer = ExerciseSessionSerializer(exercise)
        return Response(serializer.data)
    
    def delete(self, request, pk, format=None):
        exercise = self.get_object(pk)
        exercise.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ExerciseROM(APIView):
    def get(self, request):

        roms = ExerciseROMData.objects.all()
        
        # filter by exercise name
        exercise = request.query_params.get('exercise')
        if exercise is not None:
            roms = roms.filter(exercise=exercise)

        # filter by start and end date
        start_date_str = request.query_params.get('start_date')
        if start_date_str is not None:
            start = datetime.fromisoformat(start_date_str)
            roms = roms.filter(date__gte=start)

        # filter by start and end date
        end_date_str = request.query_params.get('end_date')
        if end_date_str is not None:
            end = datetime.fromisoformat(end_date_str)
            roms = roms.filter(date__lte=end)

        joint = request.query_params.get('joint')
        if joint is not None:
            roms = roms.filter(joint=joint)
        
        reference_id = request.query_params.get('reference_id')
        if reference_id is not None:
            roms = roms.filter(reference_session__pk=reference_id)

        serializer = ExerciseROMDataSerializer(roms, many=True)
        return Response(serializer.data)



class ExerciseROMDetail(APIView):
    """
    Retreive, update, or delete a reference instance
    """
    def get_object(self, pk):
        try:
            return ExerciseROMData.objects.get(pk=pk)
        except ExerciseROMData.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        reference = self.get_object(pk)
        serializer = ExerciseROMDataSerializer(reference)
        return Response(serializer.data)


class DTW(APIView):
    def post(self, request):
        payload = request.data

        first_obj = self.get_object(payload['first'])
        second_obj = self.get_object(payload['second'])

        dtw_data = rom_processing.dtw_objects(first_obj, second_obj)


        return Response(dtw_data, status=status.HTTP_200_OK)
    
    def get_object(self, obj_meta):
        ref_type = obj_meta['object_type']

        obj = None

        if ref_type == 'Reference':
            obj = ReferenceSession.objects.get(pk=obj_meta['id'])
        else:
            obj = ExerciseSession.objects.get(pk=obj_meta['id'])

        return obj
        
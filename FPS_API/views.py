# todo/todo_api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
#from rest_framework import permissions
from rest_framework.decorators import parser_classes
from rest_framework.parsers import JSONParser
from Apps.homeApp.models import DataFileUpload
#from .models import Todo
from .serializers import fpsSerializer


class csvUploadView(APIView):
    # add permission to check if user is authenticated
    #permission_classes = [permissions.IsAuthenticated]

    #parser_classes = [MultiPartParser]

    # 1. List all
    def post(self, request, filename, format=None):
        csv_obj = request.data['file']
        '''
        List all the requests items for given requested user

        '''
        DataFileUpload.objects.create(
                file_name=filename,
                actual_file=filename,
                description=request.data['description_file']
                )
                
       # csv = DataFileUpload.objects.filter(user = request.user.id)
        serializer = fpsSerializer(csv_obj, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
      
    # # 2. Create
    # def post(self, request, *args, **kwargs):
    #     '''
    #     Create the Todo with given todo data
    #     '''
    #     data = {
    #         'task': request.data.get('task'), 
    #         'completed': request.data.get('completed'), 
    #         'user': request.user.id
    #     }
    #     serializer = TodoSerializer(data=data)
    #     if serializer.is_valid():
    #         serializer.save()
    #         return Response(serializer.data, status=status.HTTP_201_CREATED)

    #     return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
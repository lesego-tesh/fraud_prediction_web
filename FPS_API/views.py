from django.shortcuts import render
from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from rest_framework import status
from django.contrib import messages
from Apps.homeApp.predModel import Prediction

from .Serializers import UploadSerializer
# Import Models
from Apps.homeApp.models import DataFileUpload

# Create your views here.
class UploadViewSet(ViewSet):
    serializer_class = UploadSerializer

    def list(self, request):
        csvs=DataFileUpload.objects.all()
        serializer = UploadSerializer(csvs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def create(self, request):
        if request.method == 'POST':
            try:
                file_uploaded = request.FILES.get('actual_file')
                content_type = file_uploaded.content_type
                data_file_name  = request.POST.get('file_name')
                description  = request.POST.get('description')
                
                DataFileUpload.objects.create(
                    file_name=data_file_name,
                    actual_file=file_uploaded,
                    description=description,
                    
                )

                response = "POST API and you have uploaded a {} file".format(content_type)
                return Response(response)
                
            except:
                response = "Invalid/wrong format. Please upload File."
                return Response(response)

class PredictionModelViewSet(ViewSet):
    serializer_class = UploadSerializer

    def list(self, request,id =None,*args, **kwargs):
        # The URL parameters are available in self.kwargs.

        context = {}
        file_obj=DataFileUpload.objects.get(id=id)
        file_loc = str(file_obj.actual_file)
        fileloc = file_loc.replace('/', '\\')
        fileloc = "media\\" + fileloc
        model = Prediction(fileloc)
        context = model.run()
        return Response(context, status=status.HTTP_200_OK)
        

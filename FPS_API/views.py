from django.shortcuts import render
from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from rest_framework import status
from django.contrib import messages
from Apps.homeApp.predModel import predfunction
from Apps.homeApp.evaluate import Evaluate
from .Serializers import UploadSerializer,transactionUploadSerializer
# Import Models
from Apps.homeApp.models import DataFileUpload,transactionUpload


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

class Evaluation(ViewSet):
    serializer_class = UploadSerializer

    def list(self, request,*args, **kwargs):
        # The URL parameters are available in self.kwargs.
        context = {}
            # print(id)
        file_obj=DataFileUpload.objects.latest('id')
        file_loc = str(file_obj.actual_file)
        fileloc = file_loc.replace('/', '\\')
        fileloc = "media\\" + fileloc
        model = Evaluate(fileloc)
        context = model.run()

        return Response(context, status=status.HTTP_200_OK)

class Predictions_upload(ViewSet):
    serializer_class = UploadSerializer

    
    def list(self, request):
        csvs=transactionUpload.objects.all()
        serializer = UploadSerializer(csvs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


    def create (self,request):
        if request.method == 'POST':
            data_file_name  = request.POST.get('data_file_name')
            try:

                file_uploaded = request.FILES.get('actual_file')
                content_type = file_uploaded.content_type
                data_file_name  = request.POST.get('file_name')
                description  = request.POST.get('description')

                t = transactionUpload.objects.create (
                file_name=data_file_name,
                actual_file= file_uploaded,
                description= description,
                
            )
                response = "POST API and you have uploaded a {} file".format(content_type)
                return Response(response)
                

            except:
                response = "Invalid/wrong format. Please upload File."
                return Response(response)

# class predict (ViewSet):
#  serializer_class = UploadSerializer
#  def list(self, request):
#             # The URL parameters are available in self.kwargs.
    
#             context = {}
#             file_obj=transactionUpload.objects.get(id)
#             file_loc = str(file_obj.actual_file)
#             fileloc = file_loc.replace('/', '\\')
#             fileloc = "media\\" + fileloc
#             print(fileloc)
#             prediction = predfunction(fileloc)
#             context['filename'] = file_loc = str(file_obj.file_name)
#             context["results"] = prediction.predict()
            
#             return Response(context, status=status.HTTP_200_OK)


   

class predictLatest(ViewSet):

        def list(self, request,*args, **kwargs):
            # The URL parameters are available in self.kwargs.

            context = {}
            file_obj=transactionUpload.objects.latest('id')
            file_loc = str(file_obj.actual_file)
            fileloc = file_loc.replace('/', '\\')
            fileloc = "media\\" + fileloc
            print(fileloc)
            prediction = predfunction(fileloc)
            context['filename'] = file_loc = str(file_obj.file_name)
            context["results"] = prediction.predict()
            
            return Response(context, status=status.HTTP_200_OK)


           
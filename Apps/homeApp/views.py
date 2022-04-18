import csv
import os
from django.shortcuts import render,redirect,get_object_or_404
from django.http.response import HttpResponseRedirect,HttpResponse
from django.contrib.auth import get_user_model, authenticate
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.conf import settings
from .models import DataFileUpload, transactionUpload
from .predModel import predfunction
def base(request):
    return render(request,'homeApp/landing_page.html')
    
def upload_credit_data(request):
    return render(request,'homeApp/upload_credit_data.html')

def predict_page(request,id):
    context = {}
    file_obj= transactionUpload.objects.get(id=id)
    file_loc = str(file_obj.actual_file)
    fileloc = file_loc.replace('/', '\\')
    fileloc = "media\\" + fileloc
    print(fileloc)
    prediction = predfunction(fileloc)
    context["results"] = prediction.predict()
    
    return render(request,'homeApp/Predict.html',context)

def predictions_page(request):
    return render(request,'homeApp/Predictions_upload.html')

def prediction_button(request,id):
    context = {}
    print(id)
    # file_obj=DataFileUpload.objects.get(id=id)
    # file_loc = str(file_obj.actual_file)
    # fileloc = file_loc.replace('/', '\\')
    # fileloc = "media\\" + fileloc
    # model = Prediction(fileloc)
    # context = model.run()
    print("\n context")
    print(context)
    
    return render(request,'homeApp/analysis.html', context)
    
def reports(request):
    all_data_files_objs=DataFileUpload.objects.all()
    return render(request,'homeApp/reports.html',{'all_files':all_data_files_objs})
    
def enter_form_data_manually(request):
    return render(request,'homeApp/enter_form_data_manually.html')
def predict_data_manually(request):
    return render(request,'homeApp/predict_data_manually.html')

def add_files_single(request,id):
    file_obj=DataFileUpload.objects.get(id=id)
    file_loc = str(file_obj.actual_file)
    fileloc = file_loc.replace('/', '\\')
    fileloc = "media\\" + fileloc
    # model = Prediction(fileloc)
    context = model.run()
    print("\n context")
    print(context)
    
    return render(request,'homeApp/analysis.html', context)
def predict_csv_single(request):
    return render(request,'homeApp/predict_csv_single.html')

def add_files_multi(request):
    return render(request,'homeApp/add_files_multi.html')
    
def predict_csv_multi(request):
    return render(request,'homeApp/predict_csv_multi.html')

def account_details(request):
    return render(request,'homeApp/account_details.html')
def change_password(request):
    return render(request,'homeApp/change_password.html')
def analysis(request):
     context = {}
     prediction = predfunction('')
     context["results"] = prediction.evaluate()
     return render(request,'homeApp/analysis.html', context)
     
def decode_utf8(input_iterator):
    for l in input_iterator:
        yield l.decode('utf-8')
def view_data(request,id):
    context = {}
    file_obj=DataFileUpload.objects.get(id=id)
    file_loc = str(file_obj.actual_file)
    fileloc = file_loc.replace('/', '\\')
    fileloc = "media\\" + fileloc
    path = os.path.join(settings.BASE_DIR, fileloc)
    f = open(path, "rb")
    reader = csv.DictReader(decode_utf8(f))
    for row in reader:
        header = list(row.keys())
        break
    data = []
    for index, row in enumerate (reader):
        values = list(row.values())
        data.append(values)
        if index == 20:
            break
    context['header'] = header
    context['data'] = data
    return render(request,'homeApp/view_data.html', context)
def delete_data(request,id):
    obj=DataFileUpload.objects.get(id=id)
    obj.delete()
    messages.success(request, "File Deleted succesfully",extra_tags = 'alert alert-success alert-dismissible show')
    return HttpResponseRedirect('/reports')
def upload_data(request):
    if request.method == 'POST':
            data_file_name  = request.POST.get('data_file_name')
            try:
                actual_file = request.FILES['actual_file_name']
                
            except:
                messages.warning(request, "Invalid/wrong format. Please upload File.")
                return redirect('/upload_credit_data')

                
            description  = request.POST.get('description')


            DataFileUpload.objects.create(
                file_name=data_file_name,
                actual_file=actual_file,
                description=description,
                
            )
            messages.success(request, "File Uploaded succesfully",extra_tags = 'alert alert-success alert-dismissible show')
            return HttpResponseRedirect('/reports')
    # return HttpResponseRedirect('reports')
def Predictions_upload(request):
    if request.method == 'POST':
            data_file_name  = request.POST.get('data_file_name')
            try:
                actual_file = request.FILES['actual_file_name']
                
            except:
                messages.warning(request, "Invalid/wrong format. Please upload File.")
                return redirect('/upload_credit_data')

                
            description  = request.POST.get('description')


            t = transactionUpload.objects.create (
                file_name=data_file_name,
                actual_file=actual_file,
                description=description,
                
            )
            messages.success(request, "File Uploaded succesfully",extra_tags = 'alert alert-success alert-dismissible show')
            return HttpResponseRedirect('/predict/'+str(t.id))
    # return HttpResponseRedirect('reports')
    

def userLogout(request):
    try:
      del request.session['username']
    except:
      pass
    logout(request)
    return HttpResponseRedirect('/') 
    

def login2(request):
    data = {}
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        print(user)
        if user:
            login(request, user)
            return HttpResponseRedirect('/')
        
        else:    
            data['error'] = "Username or Password is incorrect"
            res = render(request, 'homeApp/login.html', data)
            return res
    else:
        return render(request, 'homeApp/login.html', data)


def about(request):
    return render(request,'homeApp/about.html')

def dashboard(request):
    return render(request,'homeApp/dashboard.html')
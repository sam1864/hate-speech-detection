from django.shortcuts import render,HttpResponse
from datetime import datetime
from home.models import hatespeech
from . import ha as h
import os
import uuid
from moviepy.editor import *



# Create your views here.
def index(request):
    return render(request,'index.html')



def text(request):
    if request.method =="POST":
        if 'submit' in request.POST:
            mytext= request.POST.get('text')
            hate=""
            hate = h.haterd(str(mytext))
            Hatespeech =hatespeech(text=mytext,result=hate,date=datetime.today())
            Hatespeech.save()
            
            return render(request, "home.html",{'hate': hate, 'mytext' :mytext})
        elif 'submit1' in request.POST:
            mytext= request.POST.get('text')
            hate=""
            hate = h.haterd(str(mytext))
            pdf = h.pdfcreate(mytext,hate)
            Hatespeech =hatespeech(text=mytext,result=hate,date=datetime.today())
            Hatespeech.save()
            return render(request, "home.html",{'hate': hate, 'mytext' :mytext})
            
    return render(request, "home.html") 

def micro(request):
    if request.method =="POST":
        if 'submit' in request.POST:
            hate1,hate2=h.voice()
            if hate1==None or hate2==None:
                return render(request, "voice.html", {'alert_message': "Error: Sorry could not recognize your voice"})
            else:
                Hatespeech =hatespeech(text=hate2,result=hate1,date=datetime.today())
                Hatespeech.save()
                return render(request, "voice.html",{'hate1': hate1 , 'hate2':hate2})
        elif 'submit1' in request.POST:
            hate1,hate2=h.voice()
            if hate1==None or hate2==None:
                return render(request, "voice.html", {'alert_message': "Error: Sorry could not recognize your voice"})
            else:
                
                pdf = h.pdfcreate(hate2,hate1)
                Hatespeech =hatespeech(text=hate2,result=hate1,date=datetime.today())
                Hatespeech.save()
                return render(request, "voice.html",{'hate1': hate1 , 'hate2':hate2})
    return render(request, "voice.html") 

def link(request):
    if request.method == "POST":
        try:
            def pathfind():
                file = request.FILES['file']
                file_extension = os.path.splitext(file.name)[1]  # Get the file extension
                file_name = str(uuid.uuid4()) + file_extension  # Generate a unique file name
                file_path = os.path.join(r'enter your file location', file_name)  # Set the file path
        
                with open(file_path, 'wb') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)

                return file_path
        
            if 'submit' in request.POST:
                file_path = pathfind()           
                hate3, hate4 = h.audfie(file_path)
                Hatespeech = hatespeech(text=hate4, result=hate3, date=datetime.today())
                Hatespeech.save()
        
                return render(request, 'file.html', {'hate3': hate3, 'hate4': hate4})
            elif 'submit1' in request.POST:
                file_path = pathfind()
                hate3, hate4 = h.audfie(file_path)
                pdf = h.pdfcreate(hate4, hate3)
                Hatespeech = hatespeech(text=hate4, result=hate3, date=datetime.today())
                Hatespeech.save()
                return render(request, 'file.html', {'hate3': hate3, 'hate4': hate4})
        except Exception as e:
            # Handle the exception as per your requirements
            return render(request, 'file.html', {'alert_message': str(e)})
    else:
        return render(request, 'file.html')


def video(request):
    if request.method == "POST":
        try:
            def pathfind():
                file = request.FILES['file']
                file_extension = os.path.splitext(file.name)[1]  # Get the file extension
                file_name = str(uuid.uuid4()) + file_extension  # Generate a unique file name
                file_path = os.path.join(r'enter your file location', file_name)  # Set the file path
                with open(file_path, 'wb') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
        
                video = VideoFileClip(file_path)
                duration = video.duration
                video.close()
                clip1 = VideoFileClip(file_path).subclip(0, duration)
                file_path1 = r'enter your file location to write audio file '
                clip1.audio.write_audiofile(file_path1)
                return file_path1

            if 'submit' in request.POST: 
                file_path1 = pathfind()
                hate5, hate6 = h.audfie(file_path1)
                Hatespeech = hatespeech(text=hate6, result=hate5, date=datetime.today())
                Hatespeech.save()
                return render(request, 'video.html', {'hate5': hate5, 'hate6': hate6})
            elif 'submit1' in request.POST:
                file_path1 = pathfind()
                hate5, hate6 = h.audfie(file_path1)
                pdf = h.pdfcreate(hate6, hate5)
                Hatespeech = hatespeech(text=hate6, result=hate5, date=datetime.today())
                Hatespeech.save()
                return render(request, 'video.html', {'hate5': hate5, 'hate6': hate6})

        except Exception as e:
            # Handle the exception as per your requirements
            return render(request, 'video.html', {'alert_message': str(e)})

    else:
        return render(request, 'video.html')



def image(request):
    if request.method == "POST":
        try:
            def pathfind():
                file = request.FILES['file']
                file_extension = os.path.splitext(file.name)[1]  # Get the file extension
                file_name = str(uuid.uuid4()) + file_extension  # Generate a unique file name
                file_path = os.path.join(r'enter your file location', file_name)  # Set the file path
                with open(file_path, 'wb') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
                return file_path

            if 'submit' in request.POST: 
                file_path2 = pathfind()
                hate7, hate8 = h.imgtotext(file_path2)
                if hate8!="":
                    Hatespeech = hatespeech(text=hate8, result=hate7, date=datetime.today())
                    Hatespeech.save()
                    return render(request, 'img.html', {'hate7': hate7, 'hate8': hate8})
                else:
                    return render(request, "img.html", {'alert_message': "Image cannot recognized!!!!."})

            elif 'submit1' in request.POST:
                file_path2 = pathfind()
                hate7, hate8 = h.imgtotext(file_path2)
                if hate8!="":
                    pdf = h.pdfcreate(hate8, hate7)
                    Hatespeech = hatespeech(text=hate8, result=hate7, date=datetime.today())
                    Hatespeech.save()
                    return render(request, 'img.html', {'hate7': hate7, 'hate8': hate8})
                else:
                    return render(request, "img.html", {'alert_message': "Image cannot recognized!!!!."})

        except Exception as e:
            # Handle the exception as per your requirements
            return render(request, 'img.html', {'alert_message': str(e)})

    else:
        return render(request, 'img.html')


def filetotext(request):
    if request.method == "POST":
        try:
            def pathfind():
                file = request.FILES['file']
                file_extension = os.path.splitext(file.name)[1]  # Get the file extension
                file_name = str(uuid.uuid4()) + file_extension  # Generate a unique file name
                file_path = os.path.join(r'enter your file location', file_name)  # Set the file path
                with open(file_path, 'wb') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
                return file_path

            if 'submit' in request.POST: 
                file_path3 = pathfind()
                hate9, hate10 = h.ftotext(file_path3)
                Hatespeech = hatespeech(text=hate10, result=hate9, date=datetime.today())
                Hatespeech.save()
                return render(request, 'txtfile.html', {'hate9': hate9, 'hate10': hate10})
            elif 'submit1' in request.POST:
                file_path3 = pathfind()
                hate9, hate10 = h.ftotext(file_path3)
                pdf = h.pdfcreate(hate9, hate10)
                Hatespeech = hatespeech(text=hate10, result=hate9, date=datetime.today())
                Hatespeech.save()
                return render(request, 'txtfile.html', {'hate9': hate9, 'hate10': hate10})

        except Exception as e:
            # Handle the exception as per your requirements
            return render(request, 'txtfile.html', {'alert_message': str(e)})

    else:
        return render(request, 'txtfile.html')


def display(request):
    history = hatespeech.objects.order_by('-id')
    context = {'history': history}
    return render(request, 'display.html', context)


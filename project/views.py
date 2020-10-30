import json

import emoji as emoji
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from project.review import CommentFetch
from project.EnglishVader import sentiw
from textblob import TextBlob
from project.HindiVader import hindiSenti
from django.core.files.storage import FileSystemStorage
import googletrans
from googletrans import Translator
translator = Translator()

def index (request):
    context = {"home_page": "active"}
    template = 'home.html'
    return render(request, template, context)

def about (request):
    context = {"about_page": "active"}
    template = 'About.html'
    return render(request, template, context)
def empty(text):
    arraylist=['Review: '+text,'It Cannot Be Classified.','Please Enter a Valid Text.']
    return JsonResponse(arraylist,safe=False)
def empty1(text,lang):
    arraylist=['Review: '+text,'Language: '+lang,'It Cannot Be Classified.','Please Enter a Valid Text.']
    return JsonResponse(arraylist,safe=False)

def values(request):
    import os
    import re
    module_dir = os.path.dirname(__file__)
    print(module_dir)
    url = request.POST.get('text1', 'Sentiभावुक')
    print(url)

    if url == 'Sentiभावुक':
        myfile = request.FILES['myfile']

        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        print(uploaded_file_url)
        file_path = os.path.join(module_dir, uploaded_file_url)
        # print(file_path)
        output = os.path.join(module_dir, 'static/output.csv')
        outputCSV = open(output, 'w', encoding='utf8')
        f = open(file_path, 'r', encoding='utf8')
        for i in f:
            outputCSV.write(i)
        f.close()
    else:
        if(re.match('^(http(s)?:\/\/)?((w){3}.)?youtu(be|.be)?(\.com)?\/.+', url)):
            print(url)
            CommentFetch(url)
        else:
            output = os.path.join(module_dir, 'static/output.csv')
            outputCSV = open(output, 'w', encoding='utf8')
            outputCSV.write(url)
            outputCSV.close()

    return render(request, 'Classify.html')

def senti(request):


    text=request.POST['text']
    print("views ka",text)
    classifiedText=[]
    if(len(text)>=3):
        k=TextBlob(text)
        lang=k.detect_language()
        if(k.detect_language()=='en'):
            classifiedText = sentiw(text)
            print(classifiedText)
        elif(k.detect_language()=='hi'):
            emojis = ''.join(c for c in text if c in emoji.UNICODE_EMOJI)
            text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
            translatedText = translator.translate(text, dest="hi").text
            translatedText = translatedText +' '+ emojis
            classifiedText=hindiSenti(translatedText)
        else:
            classifiedText = empty1(text,lang)
    else:
        classifiedText = empty(text)
        print(classifiedText)
    return HttpResponse(classifiedText)


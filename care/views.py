from django.shortcuts import render
from django.http import HttpResponse
import json
import care.dnn.isBad as b
import random
import pymongo
# from .models import data

db = pymongo.MongoClient('mongodb://lib.plus:27017')

def index(request):
    return render(request, 'index.html')

def train(request):
    index = int(random.random() * db.torrent.hash.find({"name":{'$regex':"[\u4e00-\u9fa5][\u4e00-\u9fa5][\u4e00-\u9fa5][\u4e00-\u9fa5][\u4e00-\u9fa5][\u4e00-\u9fa5][\u4e00-\u9fa5]+?"}}).count())
    return render(request, 'train.html', {'text': list(db.torrent.hash.find({"name":{'$regex':"[\u4e00-\u9fa5][\u4e00-\u9fa5][\u4e00-\u9fa5][\u4e00-\u9fa5][\u4e00-\u9fa5][\u4e00-\u9fa5][\u4e00-\u9fa5]+?"}}).limit(1).skip(index))[0]['name']})

def predict(request):
    try:
        v = b.predict(request.GET.get('data'))
        print(v)
        return HttpResponse(json.dumps({request.GET.get('data'): v}, ensure_ascii=False))
    except Exception as e:
        return HttpResponse(json.dumps({'error': str(e)}, ensure_ascii=False))

def submit(request):
    try:
        text = request.POST.get('text')
        type = int(request.POST.get('type'))
        # data.objects.create(text=text, res=type)
        if type==0:
            dir = './data_set/good_data_train'
        else:
            dir = './data_set/bad_data_train'
        f = open(dir, 'a+')
        f.write('\n'+text)
        f.close()
        return HttpResponse(json.dumps({'success':'ok'}))
    except Exception as e:
        return HttpResponse(json.dumps({'error': str(e)}))
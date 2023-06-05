# Create your views here.
from django.http import JsonResponse
from django.shortcuts import render

from .model_tokenize import predict_func


def first_page(request):
    return render(request, "index.html")


def classification_func(request):
    if "POST" == request.method:
        question = request.POST.getlist('data')
        print("question", question)
        answer = predict_func(question)
        print("answer", answer)
        return JsonResponse({'answer': str(answer[0][0])})

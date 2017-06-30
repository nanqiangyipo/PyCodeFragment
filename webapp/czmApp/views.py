from django.shortcuts import render
from django.http import HttpResponse ,HttpResponseRedirect
# Create your views here.
from .forms import NameForm
from django.urls import reverse

def index(requst):
    return HttpResponse("<h1>love is sea ! hahah</h1>")

def get_name(request):
    if request.method =='POST':
        form = NameForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect(reverse('czm_test'))

    else:
        form = NameForm()

    return render(request,'czmApp/name.html',{'form':form})
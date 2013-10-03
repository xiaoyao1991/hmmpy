from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render_to_response, get_object_or_404
from record.models import Record

def index(request):
	return render_to_response('index.html')
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render_to_response, get_object_or_404
from record.models import Record

def random_start(request):
	try:
		candidate = Record.objects.select_for_update().filter(checked=False).order_by('?')[0]	#????lock
	except Exception, e:
		candidate = None

	if candidate:
		redirect_to = '/test/' + str(candidate.id)
		return HttpResponseRedirect(redirect_to)
	else:	# All done!
		return render_to_response('alldone.html')

def new_test(request, recordid):
	candidate = Record.objects.get(record_id=recordid)
	if candidate.checked:
		return render_to_response('review_page.html', {'candidate': candidate})

	return render_to_response('test_page.html', {'candidate': candidate})


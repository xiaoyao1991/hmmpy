from django.conf.urls import patterns, include, url

urlpatterns = patterns('record.views',
	url(r'^$', 'random_start'),
    url(r'^(?P<recordid>\d+)$', 'new_test' ),
)

from django.db import models

class Record(models.Model):
    record_id = models.AutoField(primary_key=True)
    pub_id = models.IntegerField()

    full_record = models.CharField(max_length=65535, blank=True)
    predicted_author_field = models.CharField(max_length=65535, blank=True)
    predicted_title_field = models.CharField(max_length=65535, blank=True)
    predicted_venue_field = models.CharField(max_length=65535, blank=True)
    predicted_year_field = models.CharField(max_length=65535, blank=True)

    correct_author_field = models.CharField(max_length=65535, blank=True)
    correct_title_field = models.CharField(max_length=65535, blank=True)
    correct_venue_field = models.CharField(max_length=65535, blank=True)
    correct_year_field = models.CharField(max_length=65535, blank=True)

    author_correct = models.NullBooleanField(default=None)
    title_correct = models.NullBooleanField(default=None)
    venue_correct = models.NullBooleanField(default=None)
    year_correct = models.NullBooleanField(default=None)

    checked = models.BooleanField(default=False)


    def __unicode__(self):
        return self.full_record


    
        

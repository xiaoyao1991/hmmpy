from django.db import models

class Cazoodler(models.Model):
	cazoodler_id = Model.AutoField(primary_key=True)
	name = Model.CharField(max_length=1000)
		

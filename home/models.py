from django.db import models

# Create your models here.
class hatespeech(models.Model):
    id = models.AutoField(primary_key=True)
    text= models.TextField() 
    result= models.TextField() 
     
    date= models.DateField()
from django import forms

class NameForm(forms.Form):
    your_name = forms.CharField(label='名字',max_length=100)
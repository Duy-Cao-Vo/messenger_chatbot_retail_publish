# Messenger chatbot

Project to create a chatbot for retail facebook page
![Homepage](https://github.com/Duy-Cao-Vo/messenger_chatbot_retail_publish/blob/master/project_image/messenger_chatbot_1.PNG)

## Run on local
```
python messenger_chatbot_aldo.py

$ run ngrok.exe http port
port: your local port
```
Then connect to your https://developers.facebook.com/apps

Past your https://yourapp on URL callback to link messenger with your app

Detail about ngrok: https://ngrok.com/download
## Run on server Heroku

```cmd
$ heroku run web app
web: gunicorn messenger_chatbot_aldo:app


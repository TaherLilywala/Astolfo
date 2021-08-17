import os
import discord
import requests
from astolphise import Astolphise
import numpy as np
import cv2
import keras.preprocessing.image

#load_dotenv()
TOKEN = 'ODc3MjM5MDk2ODE5NDQ5OTI2.YRvukg.yq4OUsmV7c_u30z1FMPhkK3eOaw'
GUILD = '877240529157509120'
a = Astolphise()
#TOKEN = 'ODc3MjM5MDk2ODE5NDQ5OTI2.YRvukg.yq4OUsmV7c_u30z1FMPhkK3eOaw'


client = discord.Client()


@client.event
async def on_ready():
    # for guild in client.guilds:
    #     if guild.name == GUILD:
    #         break

    # guild = discord.utils.find(lambda g: g.name == GUILD, client.guilds)
    guild = discord.utils.get(client.guilds, name=GUILD)
    print(
        'ready'
    )


@client.event
async def on_message(message):
    # print(message.author, client.user, type(client.user))
    # if message.author == client.user:
    #     return
    img_data = requests.get(message.attachments[0].url).content
    with open('temp/image_name.jpg', 'wb') as handler:
        handler.write(img_data)

    SIZE = 160
    img = cv2.imread('temp/image_name.jpg') 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # open cv reads images in BGR format so we have to convert it to RGB
    img = cv2.resize(img, (SIZE, SIZE)) #resizing image
    img = img.astype('float32') / 255.0

    test_img = keras.preprocessing.image.img_to_array(img)
    test_img = np.array(test_img)

    test_img = np.reshape(test_img,(-1,SIZE,SIZE,3))

    Prediction = a.predict(test_img)
        
    maxEle = max(Prediction[0])
    maxIndex = np.where(maxEle == Prediction)
    #send message using discord bot
    if maxIndex[1][0] == 0:
        await message.channel.send(message.author.mention + ' ' + 'Yay, astolpho :eggplant:')
    else:
        await message.channel.send(message.author.mention + ' ' + 'Sad')

        


client.run(TOKEN)

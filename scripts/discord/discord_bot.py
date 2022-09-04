from timeit import default_timer as timer
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import asyncio
import queue

import discord
from discord.ext import commands, tasks

from discord_config import settings

sys.path.append('.')

from ldm.simplet2i import T2I
model = T2I()
model.load_model()

loop = asyncio.get_event_loop()
queue = asyncio.Queue()

intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix = settings['prefix'], intents=intents)

@bot.event
async def on_ready():
    print('Bot is ready, oauth url: {}'.format(discord.utils.oauth_url(bot.user.id)))

@bot.command()
async def dream(ctx, *, arg):
    if ctx.author != bot.user:
        try:
            message = await ctx.reply('Your dream is queued')
            loop.call_soon_threadsafe(queue.put_nowait, dreaming(ctx, arg, message))
        except Exception as e:
            print("dream error: {}".format(e))
            await ctx.reply('Something is wrong...')

async def dreaming(ctx, arg, message):
    try:
        if ctx.author != bot.user:
            start = timer()
            
            quote_text = '{}'.format(arg)

            bot.loop.call_soon_threadsafe(bot.loop.create_task(message.edit(content='Dreaming for {}\'s `{}}`'.format(ctx.message.author.mention,quote_text))))
            outputs = model.txt2img(quote_text)
            bot.loop.call_soon_threadsafe(bot.loop.create_task(message.delete()))

            for output in outputs:
                output_file = discord.File(output[0], description = quote_text)
                bot.loop.call_soon_threadsafe(bot.loop.create_task(ctx.reply(content='Dreamt in `{}s` for {}\'s `{}`\nSeed {}'.format(timer() - start,ctx.message.author.mention,quote_text, output[1]), file=output_file)))
    except Exception as e:
        print("dreaming error: {}".format(e))
        bot.loop.call_soon_threadsafe(bot.loop.create_task(ctx.reply('Something is wrong...')))

async def async_handler():
    while True:
        task = await queue.get()
        await task
        queue.task_done()

def thread_handler():
    loop.create_task(async_handler())
    loop.run_forever()

model_thread = threading.Thread(target=thread_handler)
model_thread.start()
bot.run(settings['token'])
sys.exit(0)
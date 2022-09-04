from timeit import default_timer as timer
import sys
import threading

import asyncio

import discord
from discord.ext import commands, tasks

from discord_config import settings

sys.path.append('.')

from ldm.simplet2i import T2I
model = T2I()
model.load_model()

dreaming_loop = asyncio.get_event_loop()
dreaming_queue = asyncio.Queue()

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
            dreaming_loop.call_soon_threadsafe(dreaming_queue.put_nowait, dreaming(ctx, arg, message))
        except Exception as e:
            error_msg = 'Dream error: {}'.format(e)
            print(error_msg)
            on_bot_thread(ctx.reply(error_msg))

def noop():
    pass

def on_bot_thread(coroutine):
    bot.loop.create_task(coroutine) # add the coroutine as a task to the bot loop
    bot.loop.call_soon_threadsafe(noop) # Force the bot.loop to clear itself

async def dreaming(ctx, arg, message):
    try:
        if ctx.author != bot.user:
            start = timer()
            
            quote_text = '{}'.format(arg)

            on_bot_thread(message.edit(content='Dreaming for {}\'s `{}`'.format(ctx.message.author.mention,quote_text)))
            outputs = model.prompt2png(quote_text, 'outputs/img-samples')

            on_bot_thread(message.delete())

            for output in outputs:
                output_file = discord.File(output[0], description = quote_text)
                msg = 'Dreamt in `{}s` for {}\'s `{}`\nSeed {}'.format(timer() - start,ctx.message.author.mention,quote_text, output[1])
                on_bot_thread(ctx.reply(content=msg, file=output_file))
    except Exception as e:
        error_msg = 'Dreaming error: {}'.format(e)
        print(error_msg)
        on_bot_thread(ctx.reply(error_msg))

async def async_handler():
    while True:
        task = await dreaming_queue.get()
        await task
        dreaming_queue.task_done()

def thread_handler():
    dreaming_loop.create_task(async_handler())
    dreaming_loop.run_forever()

model_thread = threading.Thread(target=thread_handler, daemon=True)
model_thread.start()
bot.run(settings['token'])
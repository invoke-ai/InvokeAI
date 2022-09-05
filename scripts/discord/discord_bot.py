from timeit import default_timer as timer
import sys
import threading

import functools
import asyncio

import discord
from discord.ext import commands

from discord_config import settings

# regex for text parsing
import re
text_parser = re.compile("""
^
(?P<prompt>.+?)
(?:
    (?:\s+-+(?:width|w)[\s=](?P<width>\d{2,4})) |
    (?:\s+-+(?:height|h)[\s=](?P<height>\d{2,4})) |
    (?:\s+-+(?:steps)[\s=](?P<steps>\d{1,3})) |
    (?:\s+-+(?:str|strength)[\s=](?P<strength>(?:\d+(?:\.\d+)?))) |
    (?:\s+-+(?:number|n)[\s=](?P<number>\d{1})) |
    (?:\s+-+(?:seed|s)[\s=](?P<seed>\d+))
)*
$
""", re.IGNORECASE | re.VERBOSE)

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
    print('Bot is ready, oauth url: {}'.format(discord.utils.oauth_url(bot.user.id, permissions=discord.Permissions(2147609600))))

def noop():
    pass

def on_bot_thread(coroutine):
    bot.loop.create_task(coroutine) # add the coroutine as a task to the bot loop
    bot.loop.call_soon_threadsafe(noop) # Force the bot.loop to clear itself

@bot.command(
    name="dream", 
    description="Generate an image based on the given prompt"
)
async def dream(ctx: commands.Context, *, quote_text: str):
    if ctx.author != bot.user:
        try:
            if quote_text.endswith('--help'):
                message = await ctx.reply("""Dream help: 
`/dream <prompt> [--width N:512] [--height N:512] [--steps N:50] [--strength N:7.5] [--number N:1] [--seed N]`
`width`, `height`: The width and height of the image in multiples of 64. Higher numbers can crash the bot. Max: 9999
`steps`: The number of steps to do while building the image. More steps take longer but produce a higher quality image. Max: 999
`strength`: The strength of the prompt on the image. Must be >1. Higher numbers produce more strict prompt matches.
`number`: The number of images to produce. Max: 9
`seed`: The seed to use for your image. Defaults to a random number. Put in a previously used seed to get the same image, which can be then refined.
""")
            else:
                message = await ctx.reply(f'Your dream is queued. You are #{dreaming_queue.qsize()} in queue.')
                dreaming_loop.call_soon_threadsafe(dreaming_queue.put_nowait, dreaming(ctx, quote_text, message))
        except Exception as e:
            error_msg = 'Dream error: {}'.format(e)
            print(error_msg)
            await ctx.reply(error_msg)

async def dreaming(ctx: commands.Context, quote_text: str, message: discord.Message):
    try:
        if ctx.author != bot.user:
            start = timer()
            
            parsed = text_parser.match(quote_text)

            on_bot_thread(message.edit(content='Dreaming for {}\'s `{}`'.format(ctx.message.author.mention,quote_text)))

            outputs = await dreaming_loop.run_in_executor(None, functools.partial(
                model.prompt2png,
                parsed.group('prompt'), 
                'outputs/img-samples', 
                seed = parsed.group('seed'),
                iterations = None if parsed.group('number') is None else int(parsed.group('number')),
                cfg_scale = None if parsed.group('strength') is None else float(parsed.group('strength')),
                steps = None if parsed.group('steps') is None else int(parsed.group('steps')),
                height = None if parsed.group('height') is None else int(parsed.group('height')),
                width = None if parsed.group('width') is None else int(parsed.group('width'))
                ))
            
            """
            outputs = model.prompt2png(
                parsed.group('prompt'), 
                'outputs/img-samples', 
                seed = parsed.group('seed'),
                iterations = None if parsed.group('number') is None else int(parsed.group('number')),
                cfg_scale = None if parsed.group('strength') is None else float(parsed.group('strength')),
                steps = None if parsed.group('steps') is None else int(parsed.group('steps')),
                height = None if parsed.group('height') is None else int(parsed.group('height')),
                width = None if parsed.group('width') is None else int(parsed.group('width')))
            """
            #outputs = []

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
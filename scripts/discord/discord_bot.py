from timeit import default_timer as timer
import sys
import threading

import asyncio

import discord
from discord.ext import commands

from discord_config import settings

# regex for text parsing
import re
text_parser = re.compile('^(?P<prompt>.+?)(?:\s+-+(?:seed|s)[\s=](?P<seed>\d+))?$', re.IGNORECASE)

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

@bot.hybrid_command(
    name="sync_dream_commands",
    description="Sync slash commands for this bot"
)
@commands.guild_only()
async def sync_command(ctx: commands.Context) -> None:
    await bot.tree.sync(guild=discord.Object(id=ctx.guild.id))
    await ctx.send("Synced")

"""
class DreamFlags(commands.FlagConverter):
    prompt: str = commands.flag(description="The prompt to generate an image for", default=None)

@bot.hybrid_command(
    name="dream", 
    description="Generate an image based on the given prompt"
)
async def dream(ctx: commands.Context, flags: DreamFlags):
    print(f'flags: {flags}')
"""
@bot.command(
    name="dream", 
    description="Generate an image based on the given prompt"
)
async def dream(ctx: commands.Context, *, quote_text: str):
    if ctx.author != bot.user:
        try:
            message = await ctx.reply('Your dream is queued')
            dreaming_loop.call_soon_threadsafe(dreaming_queue.put_nowait, dreaming(ctx, quote_text, message))
        except Exception as e:
            error_msg = 'Dream error: {}'.format(e)
            print(error_msg)
            on_bot_thread(ctx.reply(error_msg))

#async def dreaming(ctx: commands.Context, flags: DreamFlags, message: discord.Message):
async def dreaming(ctx: commands.Context, quote_text: str, message: discord.Message):
    try:
        if ctx.author != bot.user:
            start = timer()
            
            parsed = text_parser.match(quote_text)

            on_bot_thread(message.edit(content='Dreaming for {}\'s `{}`'.format(ctx.message.author.mention,quote_text)))

            outputs = model.prompt2png(parsed.group('prompt'), 'outputs/img-samples', seed = parsed.group('seed'))
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
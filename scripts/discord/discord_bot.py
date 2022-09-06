from timeit import default_timer as timer
import sys
import threading

import functools
import asyncio
from typing import Literal, Optional

import discord
from discord import app_commands
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

@bot.hybrid_command(
    description='Syncs the Discord bot\'s commands'
)
@commands.guild_only()
@commands.is_owner()
async def sync(ctx: commands.Context, guilds: commands.Greedy[discord.Object] = None, spec: Literal["~", "*", "^"] = None) -> None:
    if not guilds:
        if spec == "~":
            synced = await ctx.bot.tree.sync(guild=ctx.guild)
        elif spec == "*":
            ctx.bot.tree.copy_global_to(guild=ctx.guild)
            synced = await ctx.bot.tree.sync(guild=ctx.guild)
        elif spec == "^":
            ctx.bot.tree.clear_commands(guild=ctx.guild)
            await ctx.bot.tree.sync(guild=ctx.guild)
            synced = []
        else:
            synced = await ctx.bot.tree.sync()
        await ctx.reply(
            f"Synced {len(synced)} commands {'globally' if spec is None else 'to the current guild.'}"
        )
        return
    ret = 0
    for guild in guilds:
        try:
            await ctx.bot.tree.sync(guild=guild)
        except discord.HTTPException:
            pass
        else:
            ret += 1
    await ctx.reply(f"Synced the tree to {ret}/{len(guilds)}.")

class DreamFlags(commands.FlagConverter, prefix = '--'):
    prompt: str = commands.Flag(description='The prompt to produce an image for', default=None)
    width: int = commands.Flag(description='The width of the image to produce, a multiple of 64. Defaults to 512. Higher numbers can fail to run. (64...2048)', default=512)
    height: int = commands.Flag(description='The height of the image to produce, a multiple of 64. Defaults to 512. Higher numbers can fail to run. (64...2048)', default=512)
    steps: int = commands.Flag(description='The number of steps. Defaults to 50. Higher numbers take longer to run. (1...128)', default=50)
    strength: float = commands.Flag(description='The strength to follow the prompt using. Defaults to 7.5. (1.01...40.0)', default=7.5)
    number: int = commands.Flag(description='The number of images to produce. More images take more time. Defaults to 1. (1...10)', default=1)
    seed: int = commands.Flag(description='The seed to use for your image. The same prompt + seed will produce the same image.', default=None)

    def __init__(self):
        self.prompt = None
        self.width = 512
        self.height = 512
        self.steps = 50
        self.strength = 7.5
        self.number = 1
        self.seed = None

@bot.hybrid_command(
    description="Generate an image based on the given prompt"
)
async def dream(ctx: commands.Context, *, quote_text: str = None, flags: DreamFlags = None):
    if flags is None:
        flags = DreamFlags()

    if ctx.author != bot.user:
        try:
            if (quote_text is not None):
                flags.prompt = quote_text

            flags.prompt = flags.prompt.strip()

            if len(flags.prompt) < 1:
                await ctx.reply(f'A prompt is required.')
            elif flags.width < 64 or flags.width > 2048:
                await ctx.reply(f'Width must be between 64 and 2048. Got `{flags.width}`.')
            elif flags.height < 64 or flags.height > 2048:
                await ctx.reply(f'Height must be between 64 and 2048. Got `{flags.height}`.')
            elif flags.steps < 1 or flags.steps > 128:
                await ctx.reply(f'Steps must be between 1 and 128. Got `{flags.steps}`.')
            elif flags.strength <= 1.0 or flags.strength > 40.0:
                await ctx.reply(f'Strength must be between 1.01 and 40.0. Got `{flags.strength}`.')
            elif flags.number < 1 or flags.number > 10:
                await ctx.reply(f'Number must be between 1 and 10. Got `{flags.number}`.')
            else:
                if dreaming_queue.qsize() <= 0:
                    message = await ctx.reply('Your dream is queued.')
                elif dreaming_queue.qsize() <= 1:
                    message = await ctx.reply('Your dream is queued. There is 1 dream ahead of you.')
                else:
                    message = await ctx.reply(f'Your dream is queued. There are {dreaming_queue.qsize()} dreams ahead of you.')

                dreaming_loop.call_soon_threadsafe(dreaming_queue.put_nowait, dreaming(ctx, flags, message))
        except Exception as e:
            error_msg = 'Dream error: {}'.format(e)
            print(error_msg)
            await ctx.reply(error_msg)

async def dreaming(ctx: commands.Context, flags: DreamFlags, message: discord.Message):
    try:
        if ctx.author != bot.user:
            start = timer()
            
            on_bot_thread(message.edit(content='Dreaming for {}\'s `{}`'.format(ctx.message.author.mention,flags.prompt)))

            outputs = await dreaming_loop.run_in_executor(None, functools.partial(
                model.prompt2png,
                flags.prompt, 
                'outputs/img-samples', 
                seed = flags.seed,
                iterations = flags.number,
                cfg_scale = flags.strength,
                steps = flags.steps,
                height = flags.height,
                width = flags.width
                ))

            on_bot_thread(message.delete())

            for output in outputs:
                output_file = discord.File(output[0], description = flags.prompt)
                msg = 'Dreamt in `{}s` for {}\'s `{}`\nSeed {}'.format(timer() - start, ctx.message.author.mention, flags.prompt, output[1])
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
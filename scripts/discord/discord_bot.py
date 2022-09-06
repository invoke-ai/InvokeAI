import os
from timeit import default_timer as timer
import sys
import threading
import uuid, tempfile

import functools
import asyncio
from typing import Literal, Optional

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

def make_temp_name(dir = tempfile.gettempdir()):
    return os.path.join(dir, str(uuid.uuid1()))

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
    prompt: str = commands.Flag(description='The prompt to produce an image for')
    width: int = commands.Flag(description='The width of the image to produce, a multiple of 64. Defaults to 512. Higher numbers can fail to run. (64...2048)', default=512)
    height: int = commands.Flag(description='The height of the image to produce, a multiple of 64. Defaults to 512. Higher numbers can fail to run. (64...2048)', default=512)
    steps: int = commands.Flag(description='The number of steps. Defaults to 50. Higher numbers take longer to run. (1...128)', default=50)
    strength: float = commands.Flag(description='The strength to follow the prompt using. Defaults to 7.5. (1.01...40.0)', default=7.5)
    number: int = commands.Flag(description='The number of images to produce. More images take more time. Defaults to 1. (1...10)', default=1)
    seed: int = commands.Flag(description='The seed to use for your image. The same prompt + seed will produce the same image.', default=None)

    img2img: discord.Attachment = commands.Flag(description='If you want to use the img2img function, attach an image to base your new image on.', default=None)
    img2img_noise: float = commands.Flag(description='The noise/unnoising to apply to the image if based on an image. 0.0 returns the same image, 1.0 returns a new image. Defaults to 0.999.', default=0.5)
    img2img_fit: bool = commands.Flag(description='Fit the image to the width/height provided. If false, the width/height of the image will be used', default=True)

    def __init__(self):
        self.prompt = None
        self.width = 512
        self.height = 512
        self.steps = 50
        self.strength = 7.5
        self.number = 1
        self.seed = None
        self.img2img = None
        self.img2img_noise = 0.75
        self.img2img_fit = True

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
            elif flags.img2img_noise < 0.0 or flags.img2img_noise > 0.999:
                await ctx.reply(f'img2img_noise must be between 0.0 and 0.999. Got `{flags.img2img_noise}`.')
            else:
                if dreaming_queue.qsize() <= 0:
                    message = await ctx.reply('Your dream is queued.')
                elif dreaming_queue.qsize() <= 1:
                    message = await ctx.reply('Your dream is queued. There is 1 dream ahead of you.')
                else:
                    message = await ctx.reply(f'Your dream is queued. There are {dreaming_queue.qsize()} dreams ahead of you.')

                img2img_filepath: str = None
                if (flags.img2img is not None):
                    img2img_filepath = make_temp_name()
                    await flags.img2img.save(img2img_filepath)

                dreaming_loop.call_soon_threadsafe(dreaming_queue.put_nowait, dreaming(ctx, flags, message, img2img_filepath))
        except Exception as e:
            error_msg = 'Dream error: {}'.format(e)
            print(error_msg)
            await ctx.reply(error_msg)

async def dreaming(ctx: commands.Context, flags: DreamFlags, message: discord.Message, img2img_filepath: str = None):
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
                width = flags.width,
                strength = flags.img2img_noise,
                init_img = img2img_filepath,
                fit = flags.img2img_fit
                ))

            # Only delete it if it wasn't a slash command
            if ctx.interaction is None:
                on_bot_thread(message.delete())
            else:
                on_bot_thread(message.edit(content='Finished dreaming for {}\'s `{}`'.format(ctx.message.author.mention,flags.prompt)))

            for output in outputs:
                output_file = discord.File(output[0], description = flags.prompt)
                embed = discord.Embed(
                    title=f'Finished dreaming',
                    description=f'Dreamt for {ctx.message.author.mention}\'s `{flags.prompt}`',
                )
                embed.set_image(url=f'attachment://{output_file.filename}')
                embed.add_field(name='Seconds taken', value=format(timer() - start, '.2f'))
                embed.add_field(name='Seed used', value=output[1])
                on_bot_thread(ctx.reply(file=output_file,embed=embed))
    except Exception as e:
        error_msg = 'Dreaming error: {}'.format(e)
        print(error_msg)
        on_bot_thread(ctx.reply(error_msg))
    finally:
        try:
            if img2img_filepath is not None and os.path.exists(img2img_filepath):
                os.remove(img2img_filepath)
        except Exception as e:
            print(f'Failed to delete image at {img2img_filepath}: {e}')

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
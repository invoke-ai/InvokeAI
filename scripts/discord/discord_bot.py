import os
from timeit import default_timer as timer
import sys
import threading
import uuid, tempfile

import functools
import asyncio
from typing import Literal

import discord
from discord.ext import commands
from discord import app_commands

import json

# config settings load
discord_config_path: str = None
if not discord_config_path or discord_config_path is None or not os.path.exists(discord_config_path):
    discord_config_path = os.path.dirname(os.path.abspath(__file__)) + '/discord_config.json'

if (not os.path.exists(discord_config_path)):
    input(f'NO CONFIG FOUND FILE: {discord_config_path}')
    sys.exit(191)

with open(discord_config_path, 'r') as config_file:
    settings = json.load(config_file)

if (not settings or settings is None):
    input(f'INVALID CONFIG FOUND FILE: {discord_config_path}')
    sys.exit(192)

# sd loader
from ldm.generate import Generate
model = Generate()
model.load_model()

# constants
_MAX_IMAGE_DIMENSION: int = 2048
_MAX_IMAGE_STEPS: int = 256
_MAX_IMAGE_NUMBER: int = 20
_MAX_DISCORD_EMBEDS: int = 10

# loop & queue setup
dreaming_loop = asyncio.get_event_loop()
dreaming_queue = asyncio.Queue()

# discord setup
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix = settings['prefix'], intents=intents)

@bot.event
async def on_ready():
    # permissions = Read Messages | Send Messages | Embed Links | Attach Files | Use Slash Commands
    print('Bot is ready, oauth url: {}'.format(discord.utils.oauth_url(bot.user.id, permissions=discord.Permissions(2147535872))))

def on_bot_thread(coroutine):
    bot.loop.create_task(coroutine) # add the coroutine to the bot.loop
    bot.loop.call_soon_threadsafe(noop) # Force the bot.loop to clear itself with a noop, so that the message gets handled asap

def make_temp_name(dir = tempfile.gettempdir()):
    return os.path.join(dir, str(uuid.uuid1()))

def noop():
    pass

@bot.hybrid_command(
    description='Syncs the Discord bot\'s slash commands'
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
    width: int = commands.Flag(description=f'The width of the image to produce, a multiple of 64. Defaults to 512. Higher numbers can fail to run. (64...{_MAX_IMAGE_DIMENSION})', default=512)
    height: int = commands.Flag(description=f'The height of the image to produce, a multiple of 64. Defaults to 512. Higher numbers can fail to run. (64...{_MAX_IMAGE_DIMENSION})', default=512)
    steps: int = commands.Flag(description=f'The number of steps. Defaults to 50. Higher numbers take longer to run. (1...{_MAX_IMAGE_STEPS})', default=50)
    cfg_scale: float = commands.Flag(description='The strength to follow the prompt using. Defaults to 7.5. (1.01...40.0)', default=7.5)
    iterations: int = commands.Flag(description=f'The number of images to produce. More images take more time. Defaults to 1. (1...{_MAX_IMAGE_NUMBER})', default=1, aliases=['number'])
    seed: int = commands.Flag(description='The seed to use for your image. The same prompt + seed will produce the same image.', default=None)
    seamless: bool = commands.Flag(description='Should the image generated tile without any seams?', default=False)

    img2img: discord.Attachment = commands.Flag(description='If you want to use the img2img function, attach an image to base your new image on.', default=None)
    img2img_mask: discord.Attachment = commands.Flag(description='If you want to use the img2img inpainting (masking) function, attach an image with a transparent area.', default=None)
    img2img_strength: float = commands.Flag(description='The noise/unnoising to apply to the image if based on an image. 0.0 returns the same image, 1.0 returns a new image. Defaults to 0.999.', default=0.5)
    img2img_fit: bool = commands.Flag(description='Fit the image to the width/height provided. If false, the width/height of the image will be used', default=True)

    def __init__(self):
        self.width = 512
        self.height = 512
        self.steps = 50
        self.cfg_scale = 7.5
        self.iterations = 1
        self.seed = None
        self.seamless = False
        self.img2img = None
        self.img2img_mask = None
        self.img2img_strength = 0.75
        self.img2img_fit = True

@bot.hybrid_command(
    description="Generate an image based on the given prompt"
)
@app_commands.describe(
    prompt='The prompt to produce an image for'
)
async def dream(ctx: commands.Context, *, prompt: str, flags: DreamFlags = None):
    # prompt is required, but not in flags so we can still use !dream

    if flags is None:
        flags = DreamFlags()

    if ctx.author != bot.user:
        try:
            if (prompt is not None):
                prompt = prompt.strip()

            if prompt is None or len(prompt) < 1:
                await ctx.reply(f'A prompt is required.')
            elif flags.width < 64 or flags.width > _MAX_IMAGE_DIMENSION:
                await ctx.reply(f'width must be between 64 and {_MAX_IMAGE_DIMENSION}. Got `{flags.width}`.')
            elif flags.height < 64 or flags.height > _MAX_IMAGE_DIMENSION:
                await ctx.reply(f'height must be between 64 and {_MAX_IMAGE_DIMENSION}. Got `{flags.height}`.')
            elif flags.steps < 1 or flags.steps > _MAX_IMAGE_STEPS:
                await ctx.reply(f'steps must be between 1 and {_MAX_IMAGE_STEPS}. Got `{flags.steps}`.')
            elif flags.cfg_scale <= 1.0 or flags.cfg_scale > 40.0:
                await ctx.reply(f'cfg_scale must be between 1.01 and 40.0. Got `{flags.cfg_scale}`.')
            elif flags.iterations < 1 or flags.iterations > _MAX_IMAGE_NUMBER:
                await ctx.reply(f'iterations must be between 1 and {_MAX_IMAGE_NUMBER}. Got `{flags.iterations}`.')
            elif flags.img2img_strength < 0.0 or flags.img2img_strength > 0.999:
                await ctx.reply(f'img2img_strength must be between 0.0 and 0.999. Got `{flags.img2img_strength}`.')
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

                img2img_mask_filepath: str = None
                if (flags.img2img_mask is not None):
                    img2img_mask_filepath = make_temp_name()
                    await flags.img2img_mask.save(img2img_mask_filepath)

                dreaming_loop.call_soon_threadsafe(dreaming_queue.put_nowait, dreaming(ctx, prompt, flags, message, img2img_filepath, img2img_mask_filepath))
        except Exception as e:
            error_msg = 'Dream error: {}'.format(e)
            print(error_msg)
            await ctx.reply(error_msg)

async def dreaming(ctx: commands.Context, prompt: str, flags: DreamFlags, message: discord.Message, img2img_filepath: str = None, img2img_mask_filepath: str = None):
    try:
        if ctx.author != bot.user:
            start = timer()
            
            on_bot_thread(message.edit(content='Dreaming for {}\'s `{}`'.format(ctx.message.author.mention,prompt)))

            outputs = await dreaming_loop.run_in_executor(None, functools.partial(
                model.prompt2png,
                prompt, 
                'outputs/img-samples', 
                seed = flags.seed,
                iterations = flags.iterations,
                cfg_scale = flags.cfg_scale,
                steps = flags.steps,
                height = flags.height,
                width = flags.width,
                strength = flags.img2img_strength,
                init_img = img2img_filepath,
                init_mask = img2img_mask_filepath,
                fit = flags.img2img_fit,
                seamless = flags.seamless
                ))

            files = []
            embeds = []
            for output in outputs:
                file = discord.File(output[0], description = prompt)
                embed = discord.Embed()
                embed.set_image(url=f'attachment://{file.filename}')
                embed.add_field(name='Prompt', value=prompt)
                embed.add_field(name='Time', value='{} seconds'.format(format(timer() - start,'.2f')))
                embed.add_field(name='Seed', value=output[1])
                
                embeds.append(embed)
                files.append(file)

            # in groups in case we've got more than 10 which is the max embeds
            for i in range(0, len(embeds), _MAX_DISCORD_EMBEDS):
                embedsChunk = embeds[i:i + _MAX_DISCORD_EMBEDS]
                filesChunk = files[i:i + _MAX_DISCORD_EMBEDS]
                on_bot_thread(ctx.reply(content=f'{ctx.message.author.mention}', files=filesChunk,embeds=embedsChunk))
                
            # Only delete it if it wasn't a slash command
            if ctx.interaction is None:
                on_bot_thread(message.delete())
            else:
                on_bot_thread(message.edit(content='Finished dreaming for {}\'s `{}`'.format(ctx.message.author.mention,prompt)))
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

        try:
            if img2img_mask_filepath is not None and os.path.exists(img2img_mask_filepath):
                os.remove(img2img_mask_filepath)
        except Exception as e:
            print(f'Failed to delete image at {img2img_mask_filepath}: {e}')

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
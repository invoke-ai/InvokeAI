---
title: Prompting-Features
---

# :octicons-command-palette-24: Prompting-Features

## **Reading Prompts from a File**

You can automate `invoke.py` by providing a text file with the prompts you want
to run, one line per prompt. The text file must be composed with a text editor
(e.g. Notepad) and not a word processor. Each line should look like what you
would type at the invoke> prompt:

```bash
"a beautiful sunny day in the park, children playing" -n4 -C10
"stormy weather on a mountain top, goats grazing" -s100
"innovative packaging for a squid's dinner" -S137038382
```

Then pass this file's name to `invoke.py` when you invoke it:

```bash
python scripts/invoke.py --from_file "/path/to/prompts.txt"
```

You may also read a series of prompts from standard input by providing
a filename of `-`. For example, here is a python script that creates a
matrix of prompts, each one varying slightly:

```bash
#!/usr/bin/env python

adjectives = ['sunny','rainy','overcast']
samplers = ['k_lms','k_euler_a','k_heun']
cfg = [7.5, 9, 11]

for adj in adjectives:
    for samp in samplers:
        for cg in cfg:
            print(f'a {adj} day -A{samp} -C{cg}')
```

Its output looks like this (abbreviated):

```bash
a sunny day -Aklms -C7.5
a sunny day -Aklms -C9
a sunny day -Aklms -C11
a sunny day -Ak_euler_a -C7.5
a sunny day -Ak_euler_a -C9
...
a overcast day -Ak_heun -C9
a overcast day -Ak_heun -C11
```

To feed it to invoke.py, pass the filename of "-"

```bash
python matrix.py | python scripts/invoke.py --from_file -
```

When the script is finished, each of the 27 combinations
of adjective, sampler and CFG will be executed.

The command-line interface provides `!fetch` and `!replay` commands
which allow you to read the prompts from a single previously-generated
image or a whole directory of them, write the prompts to a file, and
then replay them. Or you can create your own file of prompts and feed
them to the command-line client from within an interactive session.
See [Command-Line Interface](CLI.md) for details.

---

## **Negative and Unconditioned Prompts**

Any words between a pair of square brackets will instruct Stable Diffusion to
attempt to ban the concept from the generated image.

```text
this is a test prompt [not really] to make you understand [cool] how this works.
```

In the above statement, the words 'not really cool` will be ignored by Stable
Diffusion.

Here's a prompt that depicts what it does.

original prompt:

`#!bash "A fantastical translucent pony made of water and foam, ethereal, radiant, hyperalism, scottish folklore, digital painting, artstation, concept art, smooth, 8 k frostbite 3 engine, ultra detailed, art by artgerm and greg rutkowski and magali villeneuve" -s 20 -W 512 -H 768 -C 7.5 -A k_euler_a -S 1654590180`

<figure markdown>

![step1](../assets/negative_prompt_walkthru/step1.png)

</figure>

That image has a woman, so if we want the horse without a rider, we can
influence the image not to have a woman by putting [woman] in the prompt, like
this:

`#!bash "A fantastical translucent poney made of water and foam, ethereal, radiant, hyperalism, scottish folklore, digital painting, artstation, concept art, smooth, 8 k frostbite 3 engine, ultra detailed, art by artgerm and greg rutkowski and magali villeneuve [woman]" -s 20 -W 512 -H 768 -C 7.5 -A k_euler_a -S 1654590180`

<figure markdown>

![step2](../assets/negative_prompt_walkthru/step2.png)

</figure>

That's nice - but say we also don't want the image to be quite so blue. We can
add "blue" to the list of negative prompts, so it's now [woman blue]:

`#!bash "A fantastical translucent poney made of water and foam, ethereal, radiant, hyperalism, scottish folklore, digital painting, artstation, concept art, smooth, 8 k frostbite 3 engine, ultra detailed, art by artgerm and greg rutkowski and magali villeneuve [woman blue]" -s 20 -W 512 -H 768 -C 7.5 -A k_euler_a -S 1654590180`

<figure markdown>

![step3](../assets/negative_prompt_walkthru/step3.png)

</figure>

Getting close - but there's no sense in having a saddle when our horse doesn't
have a rider, so we'll add one more negative prompt: [woman blue saddle].

`#!bash "A fantastical translucent poney made of water and foam, ethereal, radiant, hyperalism, scottish folklore, digital painting, artstation, concept art, smooth, 8 k frostbite 3 engine, ultra detailed, art by artgerm and greg rutkowski and magali villeneuve [woman blue saddle]" -s 20 -W 512 -H 768 -C 7.5 -A k_euler_a -S 1654590180`

<figure markdown>

![step4](../assets/negative_prompt_walkthru/step4.png)

</figure>

!!! notes "Notes about this feature:"

    * The only requirement for words to be ignored is that they are in between a pair of square brackets.
    * You can provide multiple words within the same bracket.
    * You can provide multiple brackets with multiple words in different places of your prompt. That works just fine.
    * To improve typical anatomy problems, you can add negative prompts like `[bad anatomy, extra legs, extra arms, extra fingers, poorly drawn hands, poorly drawn feet, disfigured, out of frame, tiling, bad art, deformed, mutated]`.

---

## **Prompt Syntax Features**

The InvokeAI prompting language has the following features:

### Attention weighting

Append a word or phrase with `-` or `+`, or a weight between `0` and `2`
(`1`=default), to decrease or increase "attention" (= a mix of per-token CFG
weighting multiplier and, for `-`, a weighted blend with the prompt without the
term).

The following syntax is recognised:

- single words without parentheses: `a tall thin man picking apricots+`
- single or multiple words with parentheses:
  `a tall thin man picking (apricots)+` `a tall thin man picking (apricots)-`
  `a tall thin man (picking apricots)+` `a tall thin man (picking apricots)-`
- more effect with more symbols `a tall thin man (picking apricots)++`
- nesting `a tall thin man (picking apricots+)++` (`apricots` effectively gets
  `+++`)
- all of the above with explicit numbers `a tall thin man picking (apricots)1.1`
  `a tall thin man (picking (apricots)1.3)1.1`. (`+` is equivalent to 1.1, `++`
  is pow(1.1,2), `+++` is pow(1.1,3), etc; `-` means 0.9, `--` means pow(0.9,2),
  etc.)
- attention also applies to `[unconditioning]` so
  `a tall thin man picking apricots [(ladder)0.01]` will _very gently_ nudge SD
  away from trying to draw the man on a ladder

You can use this to increase or decrease the amount of something. Starting from
this prompt of `a man picking apricots from a tree`, let's see what happens if
we increase and decrease how much attention we want Stable Diffusion to pay to
the word `apricots`:

<figure markdown>

![an AI generated image of a man picking apricots from a tree](../assets/prompt_syntax/apricots-0.png)

</figure>

Using `-` to reduce apricot-ness:

| `a man picking apricots- from a tree`                                                                                          | `a man picking apricots-- from a tree`                                                                                                        | `a man picking apricots--- from a tree`                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| ![an AI generated image of a man picking apricots from a tree, with smaller apricots](../assets/prompt_syntax/apricots--1.png) | ![an AI generated image of a man picking apricots from a tree, with even smaller and fewer apricots](../assets/prompt_syntax/apricots--2.png) | ![an AI generated image of a man picking apricots from a tree, with very few very small apricots](../assets/prompt_syntax/apricots--3.png) |

Using `+` to increase apricot-ness:

| `a man picking apricots+ from a tree`                                                                                                      | `a man picking apricots++ from a tree`                                                                                                              | `a man picking apricots+++ from a tree`                                                                                                                     | `a man picking apricots++++ from a tree`                                                                                                                                           | `a man picking apricots+++++ from a tree`                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![an AI generated image of a man picking apricots from a tree, with larger, more vibrant apricots](../assets/prompt_syntax/apricots-1.png) | ![an AI generated image of a man picking apricots from a tree with even larger, even more vibrant apricots](../assets/prompt_syntax/apricots-2.png) | ![an AI generated image of a man picking apricots from a tree, but the man has been replaced by a pile of apricots](../assets/prompt_syntax/apricots-3.png) | ![an AI generated image of a man picking apricots from a tree, but the man has been replaced by a mound of giant melting-looking apricots](../assets/prompt_syntax/apricots-4.png) | ![an AI generated image of a man picking apricots from a tree, but the man and the leaves and parts of the ground have all been replaced by giant melting-looking apricots](../assets/prompt_syntax/apricots-5.png) |

You can also change the balance between different parts of a prompt. For
example, below is a `mountain man`:

<figure markdown>

![an AI generated image of a mountain man](../assets/prompt_syntax/mountain-man.png)

</figure>

And here he is with more mountain:

| `mountain+ man`                                | `mountain++ man`                               | `mountain+++ man`                              |
| ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| ![](../assets/prompt_syntax/mountain1-man.png) | ![](../assets/prompt_syntax/mountain2-man.png) | ![](../assets/prompt_syntax/mountain3-man.png) |

Or, alternatively, with more man:

| `mountain man+`                                | `mountain man++`                               | `mountain man+++`                              | `mountain man++++`                             |
| ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| ![](../assets/prompt_syntax/mountain-man1.png) | ![](../assets/prompt_syntax/mountain-man2.png) | ![](../assets/prompt_syntax/mountain-man3.png) | ![](../assets/prompt_syntax/mountain-man4.png) |

### Blending between prompts

- `("a tall thin man picking apricots", "a tall thin man picking pears").blend(1,1)`
- The existing prompt blending using `:<weight>` will continue to be supported -
  `("a tall thin man picking apricots", "a tall thin man picking pears").blend(1,1)`
  is equivalent to
  `a tall thin man picking apricots:1 a tall thin man picking pears:1` in the
  old syntax.
- Attention weights can be nested inside blends.
- Non-normalized blends are supported by passing `no_normalize` as an additional
  argument to the blend weights, eg
  `("a tall thin man picking apricots", "a tall thin man picking pears").blend(1,-1,no_normalize)`.
  very fun to explore local maxima in the feature space, but also easy to
  produce garbage output.

See the section below on "Prompt Blending" for more information about how this
works.

### Cross-Attention Control ('prompt2prompt')

Sometimes an image you generate is almost right, and you just want to change one
detail without affecting the rest. You could use a photo editor and inpainting
to overpaint the area, but that's a pain. Here's where `prompt2prompt` comes in
handy.

Generate an image with a given prompt, record the seed of the image, and then
use the `prompt2prompt` syntax to substitute words in the original prompt for
words in a new prompt. This works for `img2img` as well.

For example, consider the prompt `a cat.swap(dog) playing with a ball in the forest`. Normally, because of the word words interact with each other when doing a stable diffusion image generation, these two prompts would generate different compositions:
  - `a cat playing with a ball in the forest`
  - `a dog playing with a ball in the forest`

| `a cat playing with a ball in the forest` | `a dog playing with a ball in the forest` |
| --- | --- |
| img | img |


      - For multiple word swaps, use parentheses: `a (fluffy cat).swap(barking dog) playing with a ball in the forest`.
      - To swap a comma, use quotes: `a ("fluffy, grey cat").swap("big, barking dog") playing with a ball in the forest`.
- Supports options `t_start` and `t_end` (each 0-1) loosely corresponding to bloc97's `prompt_edit_tokens_start/_end` but with the math swapped to make it easier to
  intuitively understand. `t_start` and `t_end` are used to control on which steps cross-attention control should run. With the default values `t_start=0` and `t_end=1`, cross-attention control is active on every step of image generation. Other values can be used to turn cross-attention control off for part of the image generation process.
    - For example, if doing a diffusion with 10 steps for the prompt is `a cat.swap(dog, t_start=0.3, t_end=1.0) playing with a ball in the forest`, the first 3 steps will be run as `a cat playing with a ball in the forest`, while the last 7 steps will run as `a dog playing with a ball in the forest`, but the pixels that represent `dog` will be locked to the pixels that would have represented `cat` if the `cat` prompt had been used instead.
    - Conversely, for `a cat.swap(dog, t_start=0, t_end=0.7) playing with a ball in the forest`, the first 7 steps will run as `a dog playing with a ball in the forest` with the pixels that represent `dog` locked to the same pixels that would have represented `cat` if the `cat` prompt was being used instead. The final 3 steps will just run `a cat playing with a ball in the forest`.
    > For img2img, the step sequence does not start at 0 but instead at `(1.0-strength)` - so if the img2img `strength` is `0.7`, `t_start` and `t_end` must both be greater than `0.3` (`1.0-0.7`) to have any effect.

Prompt2prompt `.swap()` is not compatible with xformers, which will be temporarily disabled when doing a `.swap()` - so you should expect to use more VRAM and run slower that with xformers enabled.

The `prompt2prompt` code is based off
[bloc97's colab](https://github.com/bloc97/CrossAttentionControl).

Note that `prompt2prompt` is not currently working with the runwayML inpainting
model, and may never work due to the way this model is set up. If you attempt to
use `prompt2prompt` you will get the original image back. However, since this
model is so good at inpainting, a good substitute is to use the `clipseg` text
masking option:

```bash
invoke> a fluffy cat eating a hotdot
Outputs:
[1010] outputs/000025.2182095108.png: a fluffy cat eating a hotdog
invoke> a smiling dog eating a hotdog -I 000025.2182095108.png -tm cat
```

### Escaping parantheses () and speech marks ""

If the model you are using has parentheses () or speech marks "" as part of its
syntax, you will need to "escape" these using a backslash, so that`(my_keyword)`
becomes `\(my_keyword\)`. Otherwise, the prompt parser will attempt to interpret
the parentheses as part of the prompt syntax and it will get confused.

---

## **Prompt Blending**

You may blend together different sections of the prompt to explore the AI's
latent semantic space and generate interesting (and often surprising!)
variations. The syntax is:

```bash
blue sphere:0.25 red cube:0.75 hybrid
```

This will tell the sampler to blend 25% of the concept of a blue sphere with 75%
of the concept of a red cube. The blend weights can use any combination of
integers and floating point numbers, and they do not need to add up to 1.
Everything to the left of the `:XX` up to the previous `:XX` is used for
merging, so the overall effect is:

```bash
0.25 * "blue sphere" + 0.75 * "white duck" + hybrid
```

Because you are exploring the "mind" of the AI, the AI's way of mixing two
concepts may not match yours, leading to surprising effects. To illustrate, here
are three images generated using various combinations of blend weights. As
usual, unless you fix the seed, the prompts will give you different results each
time you run them.

<figure markdown>

### "blue sphere, red cube, hybrid"

</figure>

This example doesn't use melding at all and represents the default way of mixing
concepts.

<figure markdown>

![blue-sphere-red-cube-hyprid](../assets/prompt-blending/blue-sphere-red-cube-hybrid.png)

</figure>

It's interesting to see how the AI expressed the concept of "cube" as the four
quadrants of the enclosing frame. If you look closely, there is depth there, so
the enclosing frame is actually a cube.

<figure markdown>

### "blue sphere:0.25 red cube:0.75 hybrid"

![blue-sphere-25-red-cube-75](../assets/prompt-blending/blue-sphere-0.25-red-cube-0.75-hybrid.png)

</figure>

Now that's interesting. We get neither a blue sphere nor a red cube, but a red
sphere embedded in a brick wall, which represents a melding of concepts within
the AI's "latent space" of semantic representations. Where is Ludwig
Wittgenstein when you need him?

<figure markdown>

### "blue sphere:0.75 red cube:0.25 hybrid"

![blue-sphere-75-red-cube-25](../assets/prompt-blending/blue-sphere-0.75-red-cube-0.25-hybrid.png)

</figure>

Definitely more blue-spherey. The cube is gone entirely, but it's really cool
abstract art.

<figure markdown>

### "blue sphere:0.5 red cube:0.5 hybrid"

![blue-sphere-5-red-cube-5-hybrid](../assets/prompt-blending/blue-sphere-0.5-red-cube-0.5-hybrid.png)

</figure>

Whoa...! I see blue and red, but no spheres or cubes. Is the word "hybrid"
summoning up the concept of some sort of scifi creature? Let's find out.

<figure markdown>

### "blue sphere:0.5 red cube:0.5"

![blue-sphere-5-red-cube-5](../assets/prompt-blending/blue-sphere-0.5-red-cube-0.5.png)

</figure>

Indeed, removing the word "hybrid" produces an image that is more like what we'd
expect.

In conclusion, prompt blending is great for exploring creative space, but can be
difficult to direct. A forthcoming release of InvokeAI will feature more
deterministic prompt weighting.

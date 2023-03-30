---
title: Styles and Subjects
---

# :material-library-shelves: The Hugging Face Concepts Library and Importing Textual Inversion files

## Using Textual Inversion Files

Textual inversion (TI) files are small models that customize the output of
Stable Diffusion image generation. They can augment SD with specialized subjects
and artistic styles. They are also known as "embeds" in the machine learning
world.

Each TI file introduces one or more vocabulary terms to the SD model. These are
known in InvokeAI as "triggers." Triggers are often, but not always, denoted
using angle brackets as in "&lt;trigger-phrase&gt;". The two most common type of
TI files that you'll encounter are `.pt` and `.bin` files, which are produced by
different TI training packages. InvokeAI supports both formats, but its
[built-in TI training system](TEXTUAL_INVERSION.md) produces `.pt`.

The [Hugging Face company](https://huggingface.co/sd-concepts-library) has
amassed a large ligrary of &gt;800 community-contributed TI files covering a
broad range of subjects and styles. InvokeAI has built-in support for this
library which downloads and merges TI files automatically upon request. You can
also install your own or others' TI files by placing them in a designated
directory.

### An Example

Here are a few examples to illustrate how it works. All these images were
generated using the command-line client and the Stable Diffusion 1.5 model:

|         Japanese gardener          | Japanese gardener &lt;ghibli-face&gt; | Japanese gardener &lt;hoi4-leaders&gt; | Japanese gardener &lt;cartoona-animals&gt; |
| :--------------------------------: | :-----------------------------------: | :------------------------------------: | :----------------------------------------: |
| ![](../assets/concepts/image1.png) |  ![](../assets/concepts/image2.png)   |   ![](../assets/concepts/image3.png)   |     ![](../assets/concepts/image4.png)     |

You can also combine styles and concepts:

<figure markdown>
  | A portrait of &lt;alf&gt; in &lt;cartoona-animal&gt; style |
  | :--------------------------------------------------------: |
  | ![](../assets/concepts/image5.png)                         |
</figure>
## Using a Hugging Face Concept

!!! warning "Authenticating to HuggingFace"

    Some concepts require valid authentication to HuggingFace. Without it, they will not be downloaded
    and will be silently ignored.

    If you used an installer to install InvokeAI, you may have already set a HuggingFace token.
    If you skipped this step, you can:

    - run the InvokeAI configuration script again (if you used a manual installer): `invokeai-configure`
    - set one of the `HUGGINGFACE_TOKEN` or `HUGGING_FACE_HUB_TOKEN` environment variables to contain your token

    Finally, if you already used any HuggingFace library on your computer, you might already have a token
    in your local cache. Check for a hidden `.huggingface` directory in your home folder. If it
    contains a `token` file, then you are all set.


Hugging Face TI concepts are downloaded and installed automatically as you
require them. This requires your machine to be connected to the Internet. To
find out what each concept is for, you can browse the
[Hugging Face concepts library](https://huggingface.co/sd-concepts-library) and
look at examples of what each concept produces.

When you have an idea of a concept you wish to try, go to the command-line
client (CLI) and type a `<` character and the beginning of the Hugging Face
concept name you wish to load. Press ++tab++, and the CLI will show you all
matching concepts. You can also type `<` and hit ++tab++ to get a listing of all
~800 concepts, but be prepared to scroll up to see them all! If there is more
than one match you can continue to type and ++tab++ until the concept is
completed.

!!! example

    if you type in `<x` and hit ++tab++, you'll be prompted with the completions:

    ```py
    <xatu2>        <xatu>         <xbh>          <xi>           <xidiversity>  <xioboma>      <xuna>         <xyz>
    ```

    Now type `id` and press ++tab++. It will be autocompleted to `<xidiversity>`
    because this is a unique match.

    Finish your prompt and generate as usual. You may include multiple concept terms
    in the prompt.

If you have never used this concept before, you will see a message that the TI
model is being downloaded and installed. After this, the concept will be saved
locally (in the `models/sd-concepts-library` directory) for future use.

Several steps happen during downloading and installation, including a scan of
the file for malicious code. Should any errors occur, you will be warned and the
concept will fail to load. Generation will then continue treating the trigger
term as a normal string of characters (e.g. as literal `<ghibli-face>`).

You can also use `<concept-names>` in the WebGUI's prompt textbox. There is no
autocompletion at this time.

## Installing your Own TI Files

You may install any number of `.pt` and `.bin` files simply by copying them into
the `embeddings` directory of the InvokeAI runtime directory (usually `invokeai`
in your home directory). You may create subdirectories in order to organize the
files in any way you wish. Be careful not to overwrite one file with another.
For example, TI files generated by the Hugging Face toolkit share the named
`learned_embedding.bin`. You can use subdirectories to keep them distinct.

At startup time, InvokeAI will scan the `embeddings` directory and load any TI
files it finds there. At startup you will see messages similar to these:

```bash
>> Loading embeddings from /data/lstein/invokeai-2.3/embeddings
   | Loading v1 embedding file: style-hamunaptra
   | Loading v4 embedding file: embeddings/learned_embeds-steps-500.bin
   | Loading v2 embedding file: lfa
   | Loading v3 embedding file: easynegative
   | Loading v1 embedding file: rem_rezero
   | Loading v2 embedding file: midj-strong
   | Loading v4 embedding file: anime-background-style-v2/learned_embeds.bin
   | Loading v4 embedding file: kamon-style/learned_embeds.bin
   ** Notice: kamon-style/learned_embeds.bin was trained on a model with an incompatible token dimension: 768 vs 1024.
>> Textual inversion triggers: <anime-background-style-v2>, <easynegative>, <lfa>, <midj-strong>, <milo>, Rem3-2600, Style-Hamunaptra
```

Textual Inversion embeddings trained on version 1.X stable diffusion
models are incompatible with version 2.X models and vice-versa.

After the embeddings load, InvokeAI will print out a list of all the
recognized trigger terms. To trigger the term, include it in the
prompt exactly as written, including angle brackets if any and
respecting the capitalization.

There are at least four different embedding file formats, and each uses
a different convention for the trigger terms. In some cases, the
trigger term is specified in the file contents and may or may not be
surrounded by angle brackets. In the example above, `Rem3-2600`,
`Style-Hamunaptra`, and `<midj-strong>` were specified this way and
there is no easy way to change the term.

In other cases the trigger term is not contained within the embedding
file. In this case, InvokeAI constructs a trigger term consisting of
the base name of the file (without the file extension) surrounded by
angle brackets. In the example above `<easynegative`> is such a file
(the filename was `easynegative.safetensors`). In such cases, you can
change the trigger term simply by renaming the file.

## Further Reading

Please see [the repository](https://github.com/rinongal/textual_inversion) and
associated paper for details and limitations.

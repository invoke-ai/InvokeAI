# Cross-Attention Control - StableDiffusion

Unofficial implementation of "Prompt-to-Prompt Image Editing with Cross
Attention Control" with Stable Diffusion, the code is based on the offical
[Stable Diffusion repository](https://github.com/CompVis/stable-diffusion).

This repository reproduces the cross attention control algorithm in
"Prompt-to-Prompt Image Editing with Cross Attention Control"

## References

[Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626)  
[Compvis/stablediffusion](https://github.com/CompVis/stable-diffusion)  
[Unofficial implementation of cross attention control](https://github.com/bloc97/CrossAttentionControl)

## To do

* Implementation of controlling reweighting function through argument.
* Any resolution inference: The code is now operated in only the resolution
  512x512. Some parts are hard-coded the resolution of images.
* Modifying the code of visualization attention map: any nuber of sample images.

## Cross Attention Control

The word swapping, adding new phrase and reweighting function is implemented as below:

```python
# located in 'ldm/modules/attention.py::CrossAttention()'
def cross_attention_control(self, tattmap, sattmap=None, pmask=None, t=0, tthres=0, token_idx=[0], weights=[[1. , 1. , 1.]]):
    attn = tattmap
    sattn = sattmap

    h = 8
    bh, n, d = attn.shape

    if t>=tthres:
        """ 1. swap & ading new phrase """
        if sattmap is not None:
            bh, n, d = attn.shape
            pmask, sindices, indices = pmask
            pmask = pmask.view(1,1,-1).repeat(bh, n, 1)
            attn = (1-pmask)*attn[:,:,indices] + (pmask)*sattn[:,:,sindices]

        """ 2. reweighting """
        attn = rearrange(attn,'(b h) n d -> b h n d', h=h) # (6,8,4096,77) -> (img1(uc), img2(uc), img1(c), img1(c), img2(c), img3(c))
        num_iter = bh//(h*2) #: 3
        for k in range(len(token_idx)):
            for i in range(num_iter):
                attn[num_iter+i, :, :, token_idx[k]] *= weights[k][i]
        attn = rearrange(attn,'b h n d -> (b h) n d', h=h) # (6,8,4096,77)

    return attn
```

The mask and index are from the function in 'scripts/cross_attention/swap.py':

```python
def get_indice(model, prompts, sprompts, device="cuda"):
    """ from cross attention control(https://github.com/bloc97/CrossAttentionControl) """
    # input_ids: 49406, 1125, 539, 320, 2368, 6765, 525, 320, 11652, 49407]
    tokenizer = model.cond_stage_model.tokenizer
    tokens_length = tokenizer.model_max_length

    tokens = tokenizer(prompts[0], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
    stokens= tokenizer(sprompts[0], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
    
    p_ids = tokens.input_ids.numpy()[0]
    sp_ids = stokens.input_ids.numpy()[0]


    mask = torch.zeros(tokens_length)
    indices_target = torch.arange(tokens_length, dtype=torch.long)
    indices = torch.zeros(tokens_length, dtype=torch.long)
    
    for name, a0, a1, b0, b1 in SequenceMatcher(None, sp_ids, p_ids).get_opcodes():
        if b0 < tokens_length:
            if name == "equal" or (name == "replace" and a1-a0 == b1-b0):
                mask[b0:b1] = 1
                indices[b0:b1] = indices_target[a0:a1]
    
    mask = mask.to(device)
    indices = indices.to(device)
    indices_target = indices_target.to(device) 

    return [mask, indices, indices_target]
```

## Word swapping & Adding new phrase

![alt text](../assets/cross-attention_control/cat_tiger.png)
![alt text](../assets/cross-attention_control/cake_deco2.png)

Run the shell script 'scripts/cross_attention/swap.sh' as below:

```bash
python scripts/cross_attention/swap.py \
    --prompt "a cake with jelly beans decorations" \
    --n_samples 3 \
    --strength 0.99 \
    --sprompt "a cake with decorations" \
    --is-swap
```

If you want to get reulsts with only target prompt, remove the arguments
`is-swap` and `--sprompt`. The final shell script is written as below:

```bash
python scripts/cross_attention/swap.py \
    --prompt "a cake with jelly beans decorations" \
    --n_samples 3 \
    --strength 0.99
```

The results are save in './outputs/swap-samples'

## Reweighting

![alt text](../assets/cross-attention_control/snowy_mountain2.png)

(left to right, weights are -2, 1, 2, 3, 4, 5)  

The reweighting function is implemented, but it can't be controlled by argument.
The weight control through argument is not yet implemented.

Therefore, you should change the weight for the specific token index as below:

```python
# located in 'ldm/modules/attention.py::CrossAttention()'
def forward(self, x, context=None, scontext=None, pmask=None, time=None, mask=None):
    """
    x.shape: (6,4096,320)
    context.shape(6,77,768)
    q, k, v shape: (6, hw, 320), (6, 77, 320), (6, 77, 320)
    -> q,k,v shape: (32, hw, 40=320/8=self.head), (32, 77, 40=320/8=self.head), (32, 77, 40=320/8=self.head)

    - visualization.
    1. aggregate all attention map across the "timesteps" and "heads"
    2. Normalization divided by "max" with respecto to "each token"
    """

    h = self.heads
    if scontext == "selfattn":
        sim, attn, v = self.get_attmap(x=x, h=self.heads, context=context, mask=None)
        sattn = None
    else:
        if scontext is None:
            sim, attn, v = self.get_attmap(x=x, h=self.heads, context=context, mask=None)
            sattn = None

            """ cross attention control: only reweighting is possible. """
            """ The swap and adding new phrase do not work because, the source prompt does not exist in this case. """
            """
            ex) A photo of a house on a snowy mountain
            : for controlling "snowy":
            the token index=8.
            the weights for sample1~3 are -2, 1, 5 in this example.
            """
            attn = self.cross_attention_control(tattmap=attn, t=time, token_idx=[2], weights=[[-2., 1., 5.]] )
        else:
            x, sx = x.chunk(2)
            sim, attn, v = self.get_attmap(x=x, h=self.heads, context=context, mask=None)
            ssim, sattn, sv = self.get_attmap(x=sx, h=self.heads, context=scontext, mask=None)

            """ cross attention control """
            bh, hw, tleng = attn.shape
            attn = self.cross_attention_control(tattmap=attn, sattmap=sattn, pmask=pmask, t=time, token_idx=[0], weights=[[1., 1., 1.]] )
```

We can compare the results with different weight through this scripts:

(If you use "fixed_code", all the samples are generated with same fixed latent
vectors. For better comparison, I recommend you to utilize this argument.)

```bash
python ./scripts/swap.py\
    --prompt "A photo of a house on a snowy mountain"\
    --n_samples 3\
    --strength 0.99\
    --fixed_code
```

## Visualize Cross Attention Map

![alt text](../assets/cross-attention_control/attention.png)

Please note that visualization code

We follow the visualization cross-attention map as described in the Prompt-to-Prompt:

```python
# located in 'ldm/modules/attention.py::SpatialTransformer()'
def avg_attmap(self, attmap, token_idx=0):
    """
    num_sample(=batch_size) = 3
    uc,c = 2 #(unconditional, condiitonal)
    -> 3*2=6

    attmap.shape: similarity matrix.
    token_idx: index of token for visualizing, 77: [SOS, ...text..., EOS]
    """
    nsample2, head, hw, context_dim = attmap.shape

    attmap_sm = F.softmax(attmap.float(), dim=-1)  #F.softmax(torch.Tensor(attmap).float(), dim=-1) # (6, 8, hw, context_dim)
    att_map_sm = attmap_sm[nsample2//2:, :, :, :]  # (3, 8, hw, context_dim)
    att_map_mean = torch.mean(att_map_sm, dim=1)  # (3, hw, context_dim)

    b, hw, context_dim = att_map_mean.shape
    h = int(math.sqrt(hw))
    w = h

    return att_map_mean.view(b,h,w,context_dim)  # (3, h, w, context_dim)
```

For getting visualized cross-attention map, please run the 'scripts/cross_attention/visualize_all.sh' shell script:

```python
attenmap="/root/media/data1/sdm/attenmaps_apples_swap_orig"
sample_name="A_basket_full_of_apples_tar"
token_idx=5

for a in 1,1,0  0,0,1 0,0,2 0,0,3  2,2,1 2,2,2 2,2,3  0,2,1 0,2,2 0,2,3
do
        IFS=',' read item1 item2 item3 <<< "${a}"

        python visualize_attmap.py\
            --root ${attenmap}\
            --save_dir ./atten_${sample_name}_${token_idx}/\
            --slevel ${item1}\
            --elevel ${item2}\
            --stime 0\
            --etime 49\
            --res ${item3}\
            --token_idx ${token_idx}\
            --img_path ./outputs/swap-samples/${sample_name}.png
done

python visualize_comp.py\
    --root ./atten_${sample_name}_${token_idx}\
    --token_idx ${token_idx}
```

## Usage

Parameters in 'scripts/cross_attention/swap.sh':
| Name = Default Value | Description | Example |
|---|---|---|
| `prompt=""` | the target prompt as a string | `"a cake with jelly beans decorations"` |
| `sprompt=""` | the source prompt as a string | `"a cake with decorations"` |
| `is-swap=store_true` | if you word swap or adding new phrase with source prompt |
| `n_samples=3` | number of samples to generate, the default values is 3 now. |
| `is_get_attn=store_true` | store cross-attention map or not |
| `save_attn_dir=""` | the path that the cross-attention map will be saved in. |

Parameters in ./visualize_all.sh:
| Name = Default Value | Description | Example |
|---|---|---|
| `attenmap=""` | the path that the attention maps are saved in. It is same with "save_attn_dir" |
| `sample_name=""` | the name of samples that were generated, which were saved in ./outputs/swap-images | `"a_cake_with_jelly_beans_decorations"` |
| `token_idx=0` | the token index that we want to visualize. |

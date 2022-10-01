import numpy as np
import cv2
import os
from glob import glob
import math
import argparse
import torch
import torch.nn.functional as F

""" 이 과정 하고오자.. 너무 저장하는데 오래 걸린다 ... """
# def avg_attmap(attmap, token_idx=0):
#     """
#     numsample = 4
#     uc,c = 2
#     -> 4*2=8

#     attmap.shape: (uc(4)+c(4)=8, num_head=8, hw, 77(hw)) : cross att=77, self att=1024
#     token_idx: index of token for visualizing, 77: [SOS, ...text..., EOS]
#     """
#     attmap_sm = F.softmax(torch.Tensor(attmap).float(), dim=-1) # (8, 8, hw, context_dim)
#     att_map_sm = attmap_sm[4:, :, :, token_idx] # (4, 8, hw)
#     att_map_mean = torch.mean(att_map_sm, dim=1) # (4, hw)

#     b, hw = att_map_mean.shape
#     h = int(math.sqrt(hw))
#     w = h
#     #import pdb; pdb.set_trace()
#     return att_map_mean.view(b,h,w).unsqueeze(0) # (1, 4, h, w)

LEVEL = ['down', 'middle', 'up']

TIMES = [
    'time0001',
    'time0021',
    'time0041',
    'time0061',
    'time0081',
    'time0101',
    'time0121',
    'time0141',
    'time0161',
    'time0181',
    'time0201',
    'time0221',
    'time0241',
    'time0261',
    'time0281',
    'time0301',
    'time0321',
    'time0341',
    'time0361',
    'time0381',
    'time0401',
    'time0421',
    'time0441',
    'time0461',
    'time0481',
    'time0501',
    'time0521',
    'time0541',
    'time0561',
    'time0581',
    'time0601',
    'time0621',
    'time0641',
    'time0661',
    'time0681',
    'time0701',
    'time0721',
    'time0741',
    'time0761',
    'time0781',
    'time0801',
    'time0821',
    'time0841',
    'time0861',
    'time0881',
    'time0901',
    'time0921',
    'time0941',
    'time0961',
    'time0981',
]

RES = ['res008', 'res016', 'res032', 'res064']

# NUM_DOWN_8  = ["num008","num007"]                            # 32->8 [8,8,   16,16,    32,32]
# NUM_DOWN_16 = ["num005","num004"]                            # 32->8 [8,8,   16,16,    32,32]
# NUM_DOWN_32 = ["num002","num001"]                            # 32->8 [8,8,   16,16,    32,32]
# NUM_UP_8    = ["num003","num004","num005"] # 8->32 [8,8,8, 16,16,16, 32,32,32]
# NUM_UP_16   = ["num006","num007","num008"] # 8->32 [8,8,8, 16,16,16, 32,32,32]
# NUM_UP_32   = ["num009","num010","num011"] # 8->32 [8,8,8, 16,16,16, 32,32,32]

NUM_DOWN = [
    'num008',
    'num007',
    'num005',
    'num004',
    'num002',
    'num001',
]                            # 64->16 [16,16,   32,32,    64,64]
NUM_UP = [
    'num003',
    'num004',
    'num005',
    'num006',
    'num007',
    'num008',
    'num009',
    'num010',
    'num011',
]   # 16->64 [16,16,16, 32,32,32, 64,64,64]


parser = argparse.ArgumentParser()
parser.add_argument(
    '--root',
    type=str,
    default='/root/media/data1/sdm/attenmaps',
    help='down or middle or up',
)
parser.add_argument('--save_dir', type=str, default='./bear', help='down or middle or up')
parser.add_argument('--slevel', type=int, default=0, help='0=down, 1=middle, 2=up')
parser.add_argument('--elevel', type=int, default=0, help='0=down, 1=middle, 2=up')
parser.add_argument(
    '--stime',
    type=int,
    default=20,
    help='start timstep, 0=1, 1=21, 2=41, ..., n=stime*20+1',
)
parser.add_argument(
    '--etime', type=int, default=49, help='end timstep(0-49), 0=1, 1=21, 2=41, ...'
)
parser.add_argument('--res', type=int, default=1, help='resolution, 0=4, 1=8, 2=16, 3=32')
parser.add_argument(
    '--token_idx', type=int, default=3, help='token number for visualizing'
)

parser.add_argument(
    '--img_path',
    type=str,
    default=None,
    help='corresponding imgs saved in latent diffusion inference.',
)
parser.add_argument(
    '--is_mid', action='store_true', help='resolution, 0=4, 1=8, 2=16, 3=32'
)
# parser.add_argument( "--num",   type=str,   default=None,  help="order in the UNet" )

args = parser.parse_args()

root_dir = args.root
self_att_dir = os.path.join(root_dir, 'selfatt')
cross_att_dir = os.path.join(root_dir, 'crossatt')
os.makedirs(args.save_dir, exist_ok=True)   # make directory to save

avg_attmap_list = []
""" level """
for l in range(args.slevel, args.elevel + 1):

    if l == 1 and args.res != 0:   # middle level pass하는지 여부.
        continue

    """ time """
    for t in range(args.stime, args.etime + 1):

        """resolution and timesteps"""
        if l == 0:
            n_range = range(
                (args.res - 1) * 2, args.res * 2
            )   # NUM_DOWN[ (args.res-1)*2 :args.res*2]
            for n in n_range:
                """원래(6, 8, hw, context_dim) -> 코드에서미리저장(3, h, w, context_dim)"""
                attmap = np.load(
                    os.path.join(
                        cross_att_dir,
                        LEVEL[l]
                        + '_'
                        + TIMES[t]
                        + '_'
                        + RES[args.res]
                        + '_'
                        + NUM_DOWN[n]
                        + '.npy',
                    )
                )
                # attmap_avg = avg_attmap(attmap, args.token_idx)
                attmap_avg = (
                    torch.Tensor(attmap[:, :, :, args.token_idx]).float().unsqueeze(0)
                )   # (1,3,h,w)
        elif l == 2:
            n_range = range(
                (args.res - 1) * 3, args.res * 3
            )   # NUM_DOWN[ (args.res-1)*3 :args.res*3]
            for n in n_range:
                attmap = np.load(
                    os.path.join(
                        cross_att_dir,
                        LEVEL[l]
                        + '_'
                        + TIMES[t]
                        + '_'
                        + RES[args.res]
                        + '_'
                        + NUM_UP[n]
                        + '.npy',
                    )
                )
                # attmap_avg = avg_attmap(attmap, args.token_idx)
                attmap_avg = (
                    torch.Tensor(attmap[:, :, :, args.token_idx]).float().unsqueeze(0)
                )
        elif l == 1:
            attmap = np.load(
                os.path.join(
                    cross_att_dir, LEVEL[l] + '_' + TIMES[t] + '_' + RES[0] + '.npy'
                )
            )
            # attmap_avg = avg_attmap(attmap, args.token_idx)
            attmap_avg = (
                torch.Tensor(attmap[:, :, :, args.token_idx]).float().unsqueeze(0)
            )

        """ append current attmap """
        avg_attmap_list.append(attmap_avg)


""" concat all the attmap """
all_att_map = torch.cat(avg_attmap_list, dim=0)   # (n, 3, h, w)
sum_att_map = torch.sum(all_att_map, dim=0)   # (3, h, w)
# import pdb; pdb.set_trace()
max_val = torch.max(sum_att_map).item()

norm_att_map = sum_att_map / max_val   # (3, h, w)

b, h, w = norm_att_map.shape
for i in range(b):
    save_format = 'attmap{0}_tokidx:{1}_{2}_lev:{3}_{4}_tstep:{5}_{6}.png'.format(
        i, args.token_idx, RES[args.res], args.slevel, args.elevel, args.stime, args.etime
    )
    cv2.imwrite(
        os.path.join(args.save_dir, save_format),
        (norm_att_map[i].numpy() * 255).astype(np.uint8),
    )

if args.img_path is not None:
    #
    img = cv2.imread(args.img_path)   # (h,w*4,3)
    h, w4, c = img.shape
    print(img.shape)

    img = cv2.resize(img, (h * 3, h))
    h, w4, c = img.shape

    alpha = 0.3
    for i in range(b):
        # interpolation = cv2.INTER_NEAREST
        att_map = cv2.resize((norm_att_map[i].numpy() * 255).astype(np.uint8), (h, h))
        # import pdb; pdb.set_trace()
        img[:, i * h : (i + 1) * h, :] = alpha * img[:, i * h : (i + 1) * h, :] + (
            1 - alpha
        ) * att_map.reshape(h, h, 1)
    save_format = 'bundle_attmap_tokidx:{0}_{1}_lev:{2}_{3}_tstep:{4}_{5}.png'.format(
        args.token_idx, RES[args.res], args.slevel, args.elevel, args.stime, args.etime
    )
    cv2.imwrite(os.path.join(args.save_dir, save_format), img)
# import pdb; pdb.set_trace()

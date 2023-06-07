/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Collects ControlNet info to pass to other nodes
 */
export type ControlNetInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'controlnet';
  /**
   * The control image
   */
  image?: ImageField;
  /**
   * The ControlNet model to use
   */
  control_model?: 'lllyasviel/sd-controlnet-canny' | 'lllyasviel/sd-controlnet-depth' | 'lllyasviel/sd-controlnet-hed' | 'lllyasviel/sd-controlnet-seg' | 'lllyasviel/sd-controlnet-openpose' | 'lllyasviel/sd-controlnet-scribble' | 'lllyasviel/sd-controlnet-normal' | 'lllyasviel/sd-controlnet-mlsd' | 'lllyasviel/control_v11p_sd15_canny' | 'lllyasviel/control_v11p_sd15_openpose' | 'lllyasviel/control_v11p_sd15_seg' | 'lllyasviel/control_v11f1p_sd15_depth' | 'lllyasviel/control_v11p_sd15_normalbae' | 'lllyasviel/control_v11p_sd15_scribble' | 'lllyasviel/control_v11p_sd15_mlsd' | 'lllyasviel/control_v11p_sd15_softedge' | 'lllyasviel/control_v11p_sd15s2_lineart_anime' | 'lllyasviel/control_v11p_sd15_lineart' | 'lllyasviel/control_v11p_sd15_inpaint' | 'lllyasviel/control_v11e_sd15_shuffle' | 'lllyasviel/control_v11e_sd15_ip2p' | 'lllyasviel/control_v11f1e_sd15_tile' | 'thibaud/controlnet-sd21-openpose-diffusers' | 'thibaud/controlnet-sd21-canny-diffusers' | 'thibaud/controlnet-sd21-depth-diffusers' | 'thibaud/controlnet-sd21-scribble-diffusers' | 'thibaud/controlnet-sd21-hed-diffusers' | 'thibaud/controlnet-sd21-zoedepth-diffusers' | 'thibaud/controlnet-sd21-color-diffusers' | 'thibaud/controlnet-sd21-openposev2-diffusers' | 'thibaud/controlnet-sd21-lineart-diffusers' | 'thibaud/controlnet-sd21-normalbae-diffusers' | 'thibaud/controlnet-sd21-ade20k-diffusers' | 'CrucibleAI/ControlNetMediaPipeFace,diffusion_sd15' | 'CrucibleAI/ControlNetMediaPipeFace';
  /**
   * The weight given to the ControlNet
   */
  control_weight?: number;
  /**
   * When the ControlNet is first applied (% of total steps)
   */
  begin_step_percent?: number;
  /**
   * When the ControlNet is last applied (% of total steps)
   */
  end_step_percent?: number;
};


/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ColorField } from './ColorField';
import type { ImageField } from './ImageField';

/**
 * Generates an image using inpaint.
 */
export type InpaintInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'inpaint';
  /**
   * The prompt to generate an image from
   */
  prompt?: string;
  /**
   * The seed to use (omit for random)
   */
  seed?: number;
  /**
   * The number of steps to use to generate the image
   */
  steps?: number;
  /**
   * The width of the resulting image
   */
  width?: number;
  /**
   * The height of the resulting image
   */
  height?: number;
  /**
   * The Classifier-Free Guidance, higher values may result in a result closer to the prompt
   */
  cfg_scale?: number;
  /**
   * The scheduler to use
   */
  scheduler?: 'ddim' | 'ddpm' | 'deis' | 'lms' | 'pndm' | 'heun' | 'heun_k' | 'euler' | 'euler_k' | 'euler_a' | 'kdpm_2' | 'kdpm_2_a' | 'dpmpp_2s' | 'dpmpp_2m' | 'dpmpp_2m_k' | 'unipc';
  /**
   * The model to use (currently ignored)
   */
  model?: string;
  /**
   * Whether or not to produce progress images during generation
   */
  progress_images?: boolean;
  /**
   * The control model to use
   */
  control_model?: string;
  /**
   * The processed control image
   */
  control_image?: ImageField;
  /**
   * The input image
   */
  image?: ImageField;
  /**
   * The strength of the original image
   */
  strength?: number;
  /**
   * Whether or not the result should be fit to the aspect ratio of the input image
   */
  fit?: boolean;
  /**
   * The mask
   */
  mask?: ImageField;
  /**
   * The seam inpaint size (px)
   */
  seam_size?: number;
  /**
   * The seam inpaint blur radius (px)
   */
  seam_blur?: number;
  /**
   * The seam inpaint strength
   */
  seam_strength?: number;
  /**
   * The number of steps to use for seam inpaint
   */
  seam_steps?: number;
  /**
   * The tile infill method size (px)
   */
  tile_size?: number;
  /**
   * The method used to infill empty regions (px)
   */
  infill_method?: 'patchmatch' | 'tile' | 'solid';
  /**
   * The width of the inpaint region (px)
   */
  inpaint_width?: number;
  /**
   * The height of the inpaint region (px)
   */
  inpaint_height?: number;
  /**
   * The solid infill method color
   */
  inpaint_fill?: ColorField;
  /**
   * The amount by which to replace masked areas with latent noise
   */
  inpaint_replace?: number;
};


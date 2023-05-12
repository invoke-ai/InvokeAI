/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

/**
 * Generates an image using img2img.
 */
export type ImageToImageInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'img2img';
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
  scheduler?: 'ddim' | 'ddpm' | 'deis' | 'lms' | 'pndm' | 'heun' | 'euler' | 'euler_k' | 'euler_a' | 'kdpm_2' | 'kdpm_2_a' | 'dpmpp_2s' | 'dpmpp_2m' | 'dpmpp_2m_k' | 'unipc';
  /**
   * The model to use (currently ignored)
   */
  model?: string;
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
};


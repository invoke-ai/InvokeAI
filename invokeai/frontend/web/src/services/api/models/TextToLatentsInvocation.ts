/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { LatentsField } from './LatentsField';

/**
 * Generates latents from a prompt.
 */
export type TextToLatentsInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 't2l';
  /**
   * The prompt to generate an image from
   */
  prompt?: string;
  /**
   * The noise to use
   */
  noise?: LatentsField;
  /**
   * The number of steps to use to generate the image
   */
  steps?: number;
  /**
   * The Classifier-Free Guidance, higher values may result in a result closer to the prompt
   */
  cfg_scale?: number;
  /**
   * The scheduler to use
   */
  scheduler?: 'ddim' | 'dpmpp_2' | 'k_dpm_2' | 'k_dpm_2_a' | 'k_dpmpp_2' | 'k_euler' | 'k_euler_a' | 'k_heun' | 'k_lms' | 'plms';
  /**
   * Whether or not to generate an image that can tile without seams
   */
  seamless?: boolean;
  /**
   * The axes to tile the image on, 'x' and/or 'y'
   */
  seamless_axes?: string;
  /**
   * The model to use (currently ignored)
   */
  model?: string;
  /**
   * Whether or not to produce progress images during generation
   */
  progress_images?: boolean;
};


/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Generates an image using text2img.
 */
export type TextToImageInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'txt2img';
  /**
   * The prompt to generate an image from
   */
  prompt?: string;
  /**
   * The seed to use (-1 for a random seed)
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
   * The sampler to use
   */
  sampler_name?: 'ddim' | 'dpmpp_2' | 'k_dpm_2' | 'k_dpm_2_a' | 'k_dpmpp_2' | 'k_euler' | 'k_euler_a' | 'k_heun' | 'k_lms' | 'plms';
  /**
   * Whether or not to generate an image that can tile without seams
   */
  seamless?: boolean;
  /**
   * The model to use (currently ignored)
   */
  model?: string;
  /**
   * Whether or not to produce progress images during generation
   */
  progress_images?: boolean;
};


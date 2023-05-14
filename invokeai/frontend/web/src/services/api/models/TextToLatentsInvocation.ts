/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ConditioningField } from './ConditioningField';
import type { LatentsField } from './LatentsField';

/**
 * Generates latents from conditionings.
 */
export type TextToLatentsInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 't2l';
  /**
   * Positive conditioning for generation
   */
  positive_conditioning?: ConditioningField;
  /**
   * Negative conditioning for generation
   */
  negative_conditioning?: ConditioningField;
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
  scheduler?: 'ddim' | 'ddpm' | 'deis' | 'lms' | 'pndm' | 'heun' | 'euler' | 'euler_k' | 'euler_a' | 'kdpm_2' | 'kdpm_2_a' | 'dpmpp_2s' | 'dpmpp_2m' | 'dpmpp_2m_k' | 'unipc';
  /**
   * The model to use (currently ignored)
   */
  model?: string;
  /**
   * Whether or not to generate an image that can tile without seams
   */
  seamless?: boolean;
  /**
   * The axes to tile the image on, 'x' and/or 'y'
   */
  seamless_axes?: string;
};


/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ConditioningField } from './ConditioningField';
import type { LatentsField } from './LatentsField';

/**
 * Generates latents using latents as base image.
 */
export type LatentsToLatentsInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  type?: 'l2l';
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
  scheduler?: 'ddim' | 'dpmpp_2' | 'k_dpm_2' | 'k_dpm_2_a' | 'k_dpmpp_2' | 'k_euler' | 'k_euler_a' | 'k_heun' | 'k_lms' | 'plms';
  /**
   * The model to use (currently ignored)
   */
  model?: string;
  /**
   * The latents to use as a base image
   */
  latents?: LatentsField;
  /**
   * The strength of the latents to use
   */
  strength?: number;
};


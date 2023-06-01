/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ConditioningField } from './ConditioningField';
import type { ControlField } from './ControlField';
import type { LatentsField } from './LatentsField';

/**
 * Generates latents from conditionings.
 */
export type TextToLatentsInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
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
  scheduler?: 'ddim' | 'ddpm' | 'deis' | 'lms' | 'pndm' | 'heun' | 'heun_k' | 'euler' | 'euler_k' | 'euler_a' | 'kdpm_2' | 'kdpm_2_a' | 'dpmpp_2s' | 'dpmpp_2m' | 'dpmpp_2m_k' | 'unipc';
  /**
   * The model to use (currently ignored)
   */
  model?: string;
  /**
   * The control to use
   */
  control?: (ControlField | Array<ControlField>);
};


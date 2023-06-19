/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ConditioningField } from './ConditioningField';
import type { ControlField } from './ControlField';
import type { LatentsField } from './LatentsField';
import type { UNetField } from './UNetField';

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
  cfg_scale?: (number | Array<number>);
  /**
   * The scheduler to use
   */
  scheduler?: 'ddim' | 'ddpm' | 'deis' | 'lms' | 'lms_k' | 'pndm' | 'heun' | 'heun_k' | 'euler' | 'euler_k' | 'euler_a' | 'kdpm_2' | 'kdpm_2_a' | 'dpmpp_2s' | 'dpmpp_2s_k' | 'dpmpp_2m' | 'dpmpp_2m_k' | 'dpmpp_2m_sde' | 'dpmpp_2m_sde_k' | 'dpmpp_sde' | 'dpmpp_sde_k' | 'unipc';
  /**
   * UNet submodel
   */
  unet?: UNetField;
  /**
   * The control to use
   */
  control?: (ControlField | Array<ControlField>);
};


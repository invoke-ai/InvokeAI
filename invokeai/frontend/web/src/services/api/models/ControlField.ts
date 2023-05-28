/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

export type ControlField = {
  /**
   * processed image
   */
  image: ImageField;
  /**
   * control model used
   */
  control_model: string;
  /**
   * weight given to controlnet
   */
  control_weight: number;
  /**
   * % of total steps at which controlnet is first applied
   */
  begin_step_percent: number;
  /**
   * % of total steps at which controlnet is last applied
   */
  end_step_percent: number;
};


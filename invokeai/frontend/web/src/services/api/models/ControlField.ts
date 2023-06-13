/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageField } from './ImageField';

export type ControlField = {
  /**
   * The control image
   */
  image: ImageField;
  /**
   * The ControlNet model to use
   */
  control_model: string;
  /**
   * The weight given to the ControlNet
   */
  control_weight: number;
  /**
   * When the ControlNet is first applied (% of total steps)
   */
  begin_step_percent: number;
  /**
   * When the ControlNet is last applied (% of total steps)
   */
  end_step_percent: number;
};


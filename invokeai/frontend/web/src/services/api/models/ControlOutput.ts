/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ControlField } from './ControlField';

/**
 * node output for ControlNet info
 */
export type ControlOutput = {
  type?: 'control_output';
  /**
   * The output control image
   */
  control?: ControlField;
};


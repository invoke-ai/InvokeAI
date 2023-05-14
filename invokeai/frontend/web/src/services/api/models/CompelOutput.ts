/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ConditioningField } from './ConditioningField';

/**
 * Compel parser output
 */
export type CompelOutput = {
  type?: 'compel_output';
  /**
   * Conditioning
   */
  conditioning?: ConditioningField;
};


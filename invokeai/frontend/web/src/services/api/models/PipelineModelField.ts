/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BaseModelType } from './BaseModelType';

/**
 * Pipeline model field
 */
export type PipelineModelField = {
  /**
   * Name of the model
   */
  model_name: string;
  /**
   * Base model
   */
  base_model: BaseModelType;
};


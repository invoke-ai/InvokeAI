/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { BaseModelType } from './BaseModelType';
import type { ModelType } from './ModelType';
import type { SubModelType } from './SubModelType';

export type ModelInfo = {
  /**
   * Info to load submodel
   */
  model_name: string;
  /**
   * Base model
   */
  base_model: BaseModelType;
  /**
   * Info to load submodel
   */
  model_type: ModelType;
  /**
   * Info to load submodel
   */
  submodel?: SubModelType;
};


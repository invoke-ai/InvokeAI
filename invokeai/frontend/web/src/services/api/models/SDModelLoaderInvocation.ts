/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { PipelineModelField } from './PipelineModelField';

/**
 * Loading submodels of selected model.
 */
export type SDModelLoaderInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'sd_model_loader';
  /**
   * The model to load
   */
  model: PipelineModelField;
};


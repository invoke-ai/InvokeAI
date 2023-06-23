/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { PipelineModelField } from './PipelineModelField';

/**
 * Loads a pipeline model, outputting its submodels.
 */
export type PipelineModelLoaderInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'pipeline_model_loader';
  /**
   * The model to load
   */
  model: PipelineModelField;
};


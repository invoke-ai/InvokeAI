/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Loading submodels of selected model.
 */
export type ONNXSD1ModelLoaderInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'sd1_model_loader_onnx';
  /**
   * Model to load
   */
  model_name?: string;
};

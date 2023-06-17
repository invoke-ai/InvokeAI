/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Loading submodels of selected model.
 */
export type SD1ModelLoaderInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'sd1_model_loader';
  /**
   * Model to load
   */
  model_name?: string;
};
<<<<<<< HEAD

=======
>>>>>>> 76dd749b1 (chore: Rebuild API)

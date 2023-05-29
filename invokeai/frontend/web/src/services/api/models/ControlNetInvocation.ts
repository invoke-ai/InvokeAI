/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ControlModelField } from './ControlModelField';
import type { ImageField } from './ImageField';

/**
 * Collects ControlNet info to pass to other nodes
 */
export type ControlNetInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'controlnet';
  /**
   * image to process
   */
  image?: ImageField;
  /**
   * control model used
   */
  control_model?: ControlModelField;
  /**
   * weight given to controlnet
   */
  control_weight?: number;
  /**
   * % of total steps at which controlnet is first applied
   */
  begin_step_percent?: number;
  /**
   * % of total steps at which controlnet is last applied
   */
  end_step_percent?: number;
};

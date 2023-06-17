/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Experimental per-step parameter easing for denoising steps
 */
export type StepParamEasingInvocation = {
  /**
   * The id of this node. Must be unique among all nodes.
   */
  id: string;
  /**
   * Whether or not this node is an intermediate node.
   */
  is_intermediate?: boolean;
  type?: 'step_param_easing';
  /**
   * The easing function to use
   */
  easing?: 'Linear' | 'QuadIn' | 'QuadOut' | 'QuadInOut' | 'CubicIn' | 'CubicOut' | 'CubicInOut' | 'QuarticIn' | 'QuarticOut' | 'QuarticInOut' | 'QuinticIn' | 'QuinticOut' | 'QuinticInOut' | 'SineIn' | 'SineOut' | 'SineInOut' | 'CircularIn' | 'CircularOut' | 'CircularInOut' | 'ExponentialIn' | 'ExponentialOut' | 'ExponentialInOut' | 'ElasticIn' | 'ElasticOut' | 'ElasticInOut' | 'BackIn' | 'BackOut' | 'BackInOut' | 'BounceIn' | 'BounceOut' | 'BounceInOut';
  /**
   * number of denoising steps
   */
  num_steps?: number;
  /**
   * easing starting value
   */
  start_value?: number;
  /**
   * easing ending value
   */
  end_value?: number;
  /**
   * fraction of steps at which to start easing
   */
  start_step_percent?: number;
  /**
   * fraction of steps after which to end easing
   */
  end_step_percent?: number;
  /**
   * value before easing start
   */
  pre_start_value?: number;
  /**
   * value after easing end
   */
  post_end_value?: number;
  /**
   * include mirror of easing function
   */
  mirror?: boolean;
  /**
   * show easing plot
   */
  show_easing_plot?: boolean;
};

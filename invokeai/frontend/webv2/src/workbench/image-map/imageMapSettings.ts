/**
 * Accessors for the Image Map widget's persisted view settings, so the header
 * toggle and the view read the same shape from one place.
 */

export const getImageMapClickSelectsCluster = (values: Record<string, unknown>): boolean =>
  values.clickSelectsCluster === true;

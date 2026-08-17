/**
 * Accessors for the Image Map widget's persisted view settings, so the header
 * toggle and the view read the same shape from one place.
 */

export const getImageMapClickSelectsCluster = (values: Record<string, unknown>): boolean =>
  values.clickSelectsCluster === true;

/** Labels are drawn unless the user has turned them off. */
export const getImageMapShowClusterLabels = (values: Record<string, unknown>): boolean =>
  values.showClusterLabels !== false;

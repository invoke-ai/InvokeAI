export const getImageMapClickSelectsCluster = (values: Record<string, unknown>): boolean =>
  values.clickSelectsCluster === true;

/** Labels are drawn unless the user has turned them off. */
export const getImageMapShowClusterLabels = (values: Record<string, unknown>): boolean =>
  values.showClusterLabels !== false;

/** Lowest eps the backend will cluster with; below it the server floors the value anyway. */
export const MIN_CLUSTER_EPS = 0.01;
/** The API's own upper bound on the eps query parameter. */
export const MAX_CLUSTER_EPS = 2;

/**
 * The DBSCAN eps the user chose, or null while the server's adaptive value
 * is in use. Null is the default and the way back: clearing the control
 * hands the choice to the heuristic again.
 */
export const getImageMapClusterEps = (values: Record<string, unknown>): number | null => {
  const raw = values.clusterEps;

  // Bounded at BOTH ends against the endpoint's own range. A persisted value
  // outside it — an imported project, a hand-edited file, a future change to
  // the bounds — is sent on every refresh and 422s each one, leaving the map
  // stuck on an error with no way back through the UI.
  return typeof raw === 'number' && raw >= MIN_CLUSTER_EPS && raw <= MAX_CLUSTER_EPS ? raw : null;
};

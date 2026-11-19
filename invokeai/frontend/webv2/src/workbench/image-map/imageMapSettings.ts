import type { ResolvedValues } from '@platform/core/resolvedValues';

export const getImageMapClickSelectsCluster = (values: ResolvedValues): boolean => values.clickSelectsCluster === true;

/** Labels are drawn unless the user has turned them off. */
export const getImageMapShowClusterLabels = (values: ResolvedValues): boolean => values.showClusterLabels !== false;

/** Lowest eps the backend will cluster with; below it the server floors the value anyway. */
export const MIN_CLUSTER_EPS = 0.01;
/** The API's own upper bound on the eps query parameter. */
export const MAX_CLUSTER_EPS = 2;

/** The user's DBSCAN eps, or null (the default, and the way back) while the server's adaptive heuristic chooses. */
export const getImageMapClusterEps = (values: ResolvedValues): number | null => {
  const raw = values.clusterEps;

  // Bounded at BOTH ends by the endpoint's range. A persisted value outside it (an imported project, a hand-edited
  // file, changed bounds) is sent on every refresh and 422s each, stranding the map on an error the UI cannot leave.
  return typeof raw === 'number' && raw >= MIN_CLUSTER_EPS && raw <= MAX_CLUSTER_EPS ? raw : null;
};

/** The header's media-type trigger; one Image Map exists at a time, so its view can hand focus there by id. */
export const IMAGE_MAP_MEDIA_FILTER_TRIGGER_ID = 'image-map-media-filter-trigger';

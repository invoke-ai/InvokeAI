import type { ImageMapPoint } from './api';

export interface ClusterStats {
  clusterCount: number;
  /** Size of the biggest cluster, or 0 when nothing clustered. */
  largestCluster: number;
  /** Points DBSCAN labelled noise (cluster -1). */
  unclustered: number;
}

/**
 * The footer's one-line readout of a clustering.
 *
 * Lives beside the arithmetic so the sentence and the numbers cannot drift,
 * and so a test can assert the wording without mounting a widget.
 */
export const describeClusters = (
  stats: ClusterStats,
  t: (key: string, values: Record<string, unknown>) => string
): string => {
  const mediaPoints = (count: number) =>
    t('widgets.imageMap.mediaPoints', { count, formatted: count.toLocaleString() });

  return t('widgets.imageMap.clusterSummary', {
    clusters: stats.clusterCount.toLocaleString(),
    largest: mediaPoints(stats.largestCluster),
    unclustered: mediaPoints(stats.unclustered),
  });
};

/**
 * Cluster sizes over the points as drawn, for the footer's readout.
 *
 * Derived from the served points rather than reported alongside them so the
 * numbers cannot disagree with the map: these are exactly the points the plot
 * colours. Noise (cluster -1) is counted apart from the clusters, never as
 * one of them — an all-noise map is the shape that needs explaining, and
 * calling it "1 cluster" would hide it.
 *
 * One pass, no allocation per point: the map serves hundreds of thousands of
 * them and this runs whenever the point set changes.
 */
export const summarizeClusters = (points: readonly ImageMapPoint[]): ClusterStats => {
  const sizes = new Map<number, number>();
  let unclustered = 0;

  for (const point of points) {
    if (point.cluster < 0) {
      unclustered += 1;
      continue;
    }

    sizes.set(point.cluster, (sizes.get(point.cluster) ?? 0) + 1);
  }

  let largestCluster = 0;

  for (const size of sizes.values()) {
    largestCluster = Math.max(largestCluster, size);
  }

  return { clusterCount: sizes.size, largestCluster, unclustered };
};

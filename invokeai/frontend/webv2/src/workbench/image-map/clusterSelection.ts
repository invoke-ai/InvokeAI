import type { GalleryItemKey } from '@features/gallery/contracts';

import type { ImageMapPoint } from './api';

/** Cluster clicks select image/video keys ordered outward by distance, following PhotoMapAI proximity ordering. */

/**
 * Every member, however large: DBSCAN can put most of a huge gallery in one cluster, and a silent cap showed only the
 * part nearest the click. The map draws the whole cluster with scalar marker properties, and the gallery hydrates the
 * list a page at a time.
 */
export const collectClusterSelection = (
  points: ImageMapPoint[],
  clickedKey: GalleryItemKey
): GalleryItemKey[] | null => {
  const clicked = points.find((point) => point.key === clickedKey);

  if (!clicked || clicked.cluster < 0) {
    return null;
  }

  return points
    .filter((point) => point.cluster === clicked.cluster)
    .map((point) => ({
      distance: (point.x - clicked.x) ** 2 + (point.y - clicked.y) ** 2,
      key: point.key,
    }))
    .sort((left, right) => left.distance - right.distance)
    .map((entry) => entry.key);
};

// The map widget is English-only, so its numbers are grouped to match its words
// rather than by whatever locale the browser reports.
const CLUSTER_SIZE_FORMAT = new Intl.NumberFormat('en-US');

/** A cluster's size as the map words it; also the selection's name while the cluster has no label. */
export const formatClusterSize = (count: number): string =>
  `${CLUSTER_SIZE_FORMAT.format(count)} ${count === 1 ? 'item' : 'items'}`;

/**
 * Whether a selection's name is only its size, as it is for a cluster that had no label. The count in such a name was
 * taken at click time and goes stale once members are deleted, so the name is recognised rather than compared.
 */
export const isClusterSizeLabel = (label: string): boolean => /^\d{1,3}(,\d{3})* items?$/.test(label);

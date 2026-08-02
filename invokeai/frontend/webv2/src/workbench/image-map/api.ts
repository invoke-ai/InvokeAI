import { apiFetchJson } from '@platform/transport/http';

/** Mirrors the backend's ImageMapState literal. */
export type ImageMapState = 'disabled' | 'model_missing' | 'empty' | 'computing' | 'ready';

export interface ImageMapPoint {
  x: number;
  y: number;
  imageName: string;
  /** DBSCAN cluster label; -1 means unclustered noise. */
  cluster: number;
}

export interface ImageMapPoints {
  points: ImageMapPoint[];
  state: ImageMapState;
  /** The accessible image set changed since this projection was computed; a refresh is pending. */
  stale: boolean;
  pointCount: number;
  /** The configured embedding model's name; only set when state is 'model_missing'. */
  modelName: string | null;
  /** The effective DBSCAN eps these points were clustered with; pass it back to reproduce the clustering. */
  clusterEps: number | null;
  /** Fingerprint of the visible image set; label responses with a different hash were clustered over a drifted set. */
  visibleHash: string | null;
  updatedAt: string | null;
}

interface BackendImageMapPoint {
  x: number;
  y: number;
  image_name: string;
  cluster: number;
}

interface BackendImageMapPointsResponse {
  points: BackendImageMapPoint[];
  state: ImageMapState;
  stale: boolean;
  point_count: number;
  model_name?: string | null;
  cluster_eps?: number | null;
  visible_hash?: string | null;
  updated_at: string | null;
}

const mapPoints = (body: BackendImageMapPointsResponse): ImageMapPoints => ({
  clusterEps: body.cluster_eps ?? null,
  modelName: body.model_name ?? null,
  pointCount: body.point_count,
  points: body.points.map((point) => ({
    cluster: point.cluster,
    imageName: point.image_name,
    x: point.x,
    y: point.y,
  })),
  stale: body.stale,
  state: body.state,
  updatedAt: body.updated_at ?? null,
  visibleHash: body.visible_hash ?? null,
});

export const fetchImageMapPoints = async (options?: { eps?: number; minSamples?: number }): Promise<ImageMapPoints> => {
  const query = new URLSearchParams();

  if (options?.eps !== undefined) {
    query.set('eps', String(options.eps));
  }

  if (options?.minSamples !== undefined) {
    query.set('min_samples', String(options.minSamples));
  }

  // Not URLSearchParams.size: it is newer than the build's browser baseline
  // (undefined on Safari 16, which would silently drop the params).
  const queryString = query.toString();
  const body = await apiFetchJson<BackendImageMapPointsResponse>(
    `/api/v1/image_map/points${queryString ? `?${queryString}` : ''}`
  );

  return mapPoints(body);
};

export const requestImageMapRefresh = async (): Promise<boolean> => {
  const body = await apiFetchJson<{ enqueued: boolean }>('/api/v1/image_map/refresh', { method: 'POST' });

  return body.enqueued;
};

export interface ImageMapClusterLabels {
  labels: Record<string, string>;
  /** Fingerprint of the visible image set the labels were clustered over. */
  visibleHash: string | null;
  /** The projection these labels were computed against. */
  updatedAt: string | null;
}

export const fetchImageMapClusterLabels = async (options?: {
  eps?: number;
  minSamples?: number;
}): Promise<ImageMapClusterLabels> => {
  const query = new URLSearchParams();

  if (options?.eps !== undefined) {
    query.set('eps', String(options.eps));
  }

  if (options?.minSamples !== undefined) {
    query.set('min_samples', String(options.minSamples));
  }

  const queryString = query.toString();
  const body = await apiFetchJson<{
    labels: Record<string, { label: string }>;
    visible_hash?: string | null;
    updated_at?: string | null;
  }>(`/api/v1/image_map/cluster_labels${queryString ? `?${queryString}` : ''}`);

  return {
    labels: Object.fromEntries(Object.entries(body.labels).map(([clusterId, info]) => [clusterId, info.label])),
    updatedAt: body.updated_at ?? null,
    visibleHash: body.visible_hash ?? null,
  };
};

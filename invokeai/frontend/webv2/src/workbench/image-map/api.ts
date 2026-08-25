import { apiFetchJson } from '@platform/transport/http';

import type { ImageIndexCounts } from './indexProgress';

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

export interface ImageMapStatus {
  /** Embedding-index counts, or null: the backend omits them for non-admins. */
  index: ImageIndexCounts | null;
}

interface BackendImageMapStatusResponse {
  enabled: boolean;
  index?: { total: number; embedded: number; failed?: number } | null;
}

/**
 * Current index/projection status. Only the index counts are read here: the
 * projection half of the response duplicates what /points already carries, and
 * the counts are the one thing no other call returns.
 */
export const fetchImageMapStatus = async (): Promise<ImageMapStatus> => {
  const body = await apiFetchJson<BackendImageMapStatusResponse>('/api/v1/image_map/status');

  if (!body.index) {
    return { index: null };
  }

  const { embedded, total } = body.index;
  const failed = body.index.failed ?? 0;

  return {
    // `pending` is a computed property on the backend model and so is absent
    // from the serialized response; it is derived the same way here (failures
    // are excluded so it can still drain to zero).
    index: { embedded, failed, pending: Math.max(0, total - embedded - failed), total },
  };
};

export const requestImageMapRefresh = async (): Promise<boolean> => {
  const body = await apiFetchJson<{ enqueued: boolean }>('/api/v1/image_map/refresh', { method: 'POST' });

  return body.enqueued;
};

export interface ImageMapClusterLabelInfo {
  /** Best-matching vocabulary phrase. */
  label: string;
  /** Runner-up phrases, best first. */
  alternates: string[];
}

export interface ImageMapClusterLabels {
  labels: Record<string, ImageMapClusterLabelInfo>;
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
    labels: Record<string, { label: string; alternates?: string[] }>;
    visible_hash?: string | null;
    updated_at?: string | null;
  }>(`/api/v1/image_map/cluster_labels${queryString ? `?${queryString}` : ''}`);

  return {
    labels: Object.fromEntries(
      Object.entries(body.labels).map(([clusterId, info]) => [
        clusterId,
        { alternates: info.alternates ?? [], label: info.label },
      ])
    ),
    updatedAt: body.updated_at ?? null,
    visibleHash: body.visible_hash ?? null,
  };
};

export interface ImageMapImageLabels {
  /** Best-matching vocabulary phrase. */
  label: string;
  /** Runner-up phrases, best first. */
  alternates: string[];
}

export const fetchImageMapImageLabels = async (imageName: string): Promise<ImageMapImageLabels> => {
  const query = new URLSearchParams({ image_name: imageName });
  const body = await apiFetchJson<{ label: string; alternates?: string[] }>(
    `/api/v1/image_map/image_labels?${query.toString()}`
  );

  return { alternates: body.alternates ?? [], label: body.label };
};

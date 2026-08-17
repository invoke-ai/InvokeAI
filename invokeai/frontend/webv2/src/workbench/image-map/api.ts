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
  updated_at: string | null;
}

const mapPoints = (body: BackendImageMapPointsResponse): ImageMapPoints => ({
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

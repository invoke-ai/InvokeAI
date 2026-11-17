import type { GalleryItemKey } from './items';

export type GallerySemanticQuery =
  | { kind: 'text'; query: string }
  | { kind: 'image'; imageName: string }
  | { kind: 'url'; url: string }
  | { kind: 'file'; fileId: string }
  | { kind: 'cluster'; clusterId: string };

export type GallerySemanticReference =
  | { kind: 'text'; query: string }
  | { kind: 'image'; imageName: string }
  | { kind: 'url'; url: string }
  | { kind: 'file'; fileId: string; label: string }
  | { kind: 'cluster'; clusterId: string; label: string };

/** A file reference is identified by its registry key alone: relabeling must not read as a different query. */
export const gallerySemanticReferenceKey = (reference: GallerySemanticReference | null): string => {
  if (reference === null) {
    return '';
  }

  switch (reference.kind) {
    case 'text':
      return `text:${reference.query}`;
    case 'image':
      return `image:${reference.imageName}`;
    case 'url':
      return `url:${reference.url}`;
    case 'file':
      return `file:${reference.fileId}`;
    case 'cluster':
      return `cluster:${reference.clusterId}`;
  }
};

export const toGallerySemanticTextReference = (text: string): GallerySemanticReference | null => {
  const query = text.trim();

  return query ? { kind: 'text', query } : null;
};

export const toGallerySemanticQuery = (reference: GallerySemanticReference): GallerySemanticQuery => {
  switch (reference.kind) {
    case 'text':
      return { kind: 'text', query: reference.query };
    case 'image':
      return { imageName: reference.imageName, kind: 'image' };
    case 'url':
      return { kind: 'url', url: reference.url };
    case 'file':
      return { fileId: reference.fileId, kind: 'file' };
    case 'cluster':
      return { clusterId: reference.clusterId, kind: 'cluster' };
  }
};

let nextId = 0;

const files = new Map<string, { blob: Blob; label: string }>();

export const registerExternalImageFile = (blob: Blob, label: string): string => {
  nextId += 1;
  // The id is persisted, so the random token keeps ids from another realm (tab, reload) from colliding.
  const fileId = `external-${String(nextId)}-${Math.random().toString(36).slice(2, 10)}`;

  files.clear();
  files.set(fileId, { blob, label });

  return fileId;
};

export const getExternalImageFile = (fileId: string): { blob: Blob; label: string } | null => files.get(fileId) ?? null;

let nextClusterId = 0;

const clusters = new Map<string, { itemKeys: GalleryItemKey[]; label: string }>();

export const registerImageCluster = (itemKeys: GalleryItemKey[], label: string): string => {
  nextClusterId += 1;
  // Per-realm random tokens keep persisted keys from resolving to unrelated clusters after reload or in another tab.
  const clusterId = `cluster-${String(nextClusterId)}-${Math.random().toString(36).slice(2, 10)}`;

  clusters.clear();
  clusters.set(clusterId, { itemKeys, label });

  return clusterId;
};

export const getImageCluster = (clusterId: string): { itemKeys: GalleryItemKey[]; label: string } | null =>
  clusters.get(clusterId) ?? null;

/** Rollback restores only this call's removals, and only while the same registration owns the slot. */
export const pruneImageClusterMembers = (itemKeys: readonly GalleryItemKey[]): (() => void) => {
  const entry = [...clusters.entries()].at(0);

  if (!entry || itemKeys.length === 0) {
    return () => undefined;
  }

  const [clusterId, cluster] = entry;
  const requested = new Set(itemKeys);
  const originalKeys = cluster.itemKeys;
  const removedKeys = originalKeys.filter((key) => requested.has(key));

  if (removedKeys.length === 0) {
    return () => undefined;
  }

  const removed = new Set(removedKeys);

  clusters.set(clusterId, {
    itemKeys: originalKeys.filter((key) => !removed.has(key)),
    label: cluster.label,
  });

  return () => {
    const current = clusters.get(clusterId);

    if (!current) {
      return;
    }

    // Preserves concurrent prunes and rollbacks.
    const restored = new Set([...current.itemKeys, ...removedKeys]);

    clusters.set(clusterId, {
      itemKeys: originalKeys.filter((key) => restored.has(key)),
      label: current.label,
    });
  };
};

/** Their ranking positions cannot survive reload or other tabs. */
const isSessionScopedGallerySemanticReference = (value: unknown): boolean =>
  !!value &&
  typeof value === 'object' &&
  ((value as Record<string, unknown>).kind === 'file' || (value as Record<string, unknown>).kind === 'cluster');

/** Positions tied to rejected rankings would become unrelated board-page offsets. */
const stripGallerySearchPositions = (
  values: Record<string, unknown>,
  shouldStrip: (value: unknown) => boolean
): Record<string, unknown> | null => {
  if (!shouldStrip(values.semanticImageQuery)) {
    return null;
  }

  const nextValues: Record<string, unknown> = { ...values, semanticImageQuery: null };

  if (typeof values.galleryPage === 'number' && values.galleryPage !== 0) {
    nextValues.galleryPage = 0;
  }

  if (typeof values.selectedImagePage === 'number' && values.selectedImagePage !== 0) {
    nextValues.selectedImagePage = 0;
  }

  const selectedImageQuery =
    values.selectedImageQuery && typeof values.selectedImageQuery === 'object'
      ? (values.selectedImageQuery as Record<string, unknown>)
      : null;

  if (selectedImageQuery && typeof selectedImageQuery.page === 'number' && selectedImageQuery.page !== 0) {
    nextValues.selectedImageQuery = { ...selectedImageQuery, page: 0 };
  }

  return nextValues;
};

/** Users must not be stranded mid-board without pagination; real pages stay. */
export const stripInfiniteWindowAnchor = (values: Record<string, unknown>): Record<string, unknown> | null =>
  values.paginationMode !== 'paginated' && typeof values.galleryPage === 'number' && values.galleryPage > 0
    ? { ...values, galleryPage: 0 }
    : null;

/** Even one that resolves now: another realm cannot use it. */
export const stripSessionScopedGallerySearch = (values: Record<string, unknown>): Record<string, unknown> | null =>
  stripGallerySearchPositions(
    values,
    (value) =>
      isSessionScopedGallerySemanticReference(value) ||
      (value !== null && value !== undefined && parseGallerySemanticReference(value) === null)
  );

/** Only unresolved ones: same-session reopen, fork or import can keep a live ranking. */
export const stripUnresolvableGallerySearch = (values: Record<string, unknown>): Record<string, unknown> | null =>
  stripGallerySearchPositions(
    values,
    (value) => value !== null && value !== undefined && parseGallerySemanticReference(value) === null
  );

/** Tolerates the legacy bare-image-name shape; an unresolved file key reads as no search. */
export const parseGallerySemanticReference = (value: unknown): GallerySemanticReference | null => {
  if (typeof value === 'string' && value) {
    return { imageName: value, kind: 'image' };
  }

  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;

    if (record.kind === 'text' && typeof record.query === 'string' && record.query) {
      return { kind: 'text', query: record.query };
    }

    if (record.kind === 'image' && typeof record.imageName === 'string' && record.imageName) {
      return { imageName: record.imageName, kind: 'image' };
    }

    if (record.kind === 'url' && typeof record.url === 'string' && record.url) {
      return { kind: 'url', url: record.url };
    }

    if (record.kind === 'file' && typeof record.fileId === 'string' && record.fileId) {
      if (getExternalImageFile(record.fileId) === null) {
        return null;
      }

      return {
        fileId: record.fileId,
        kind: 'file',
        label: typeof record.label === 'string' && record.label ? record.label : 'dropped image',
      };
    }

    if (record.kind === 'cluster' && typeof record.clusterId === 'string' && record.clusterId) {
      const cluster = getImageCluster(record.clusterId);

      if (cluster === null) {
        return null;
      }

      return {
        clusterId: record.clusterId,
        kind: 'cluster',
        label: typeof record.label === 'string' && record.label ? record.label : cluster.label,
      };
    }
  }

  return null;
};

/** Matches this app's own image URLs so in-app drags become by-name queries. */
const APP_IMAGE_PATH = /\/api\/v\d+\/images\/i\/([^/]+)\//;

/** This app's image URLs resolve by name to avoid private-address download rejection. */
export const semanticReferenceFromDataTransfer = (dataTransfer: {
  files: ArrayLike<File>;
  getData: (format: string) => string;
}): GallerySemanticReference | null => {
  const file = Array.from(dataTransfer.files).find((candidate) => candidate.type.startsWith('image/'));

  if (file) {
    const label = file.name || 'dropped image';

    return { fileId: registerExternalImageFile(file, label), kind: 'file', label };
  }

  const uri = dataTransfer
    .getData('text/uri-list')
    .split('\n')
    .map((line) => line.trim())
    .find((line) => line && !line.startsWith('#'));

  if (!uri || !/^https?:\/\//i.test(uri)) {
    return null;
  }

  const appImageMatch = APP_IMAGE_PATH.exec(uri);

  if (appImageMatch?.[1]) {
    try {
      return { imageName: decodeURIComponent(appImageMatch[1]), kind: 'image' };
    } catch {
      return null;
    }
  }

  return { kind: 'url', url: uri };
};

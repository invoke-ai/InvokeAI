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

/**
 * Stable identity for equality checks and fetch keys. A file reference is
 * identified by its registry key alone — the label is presentation, not
 * identity, so relabeling must not read as a different query.
 */
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

/** Blank or whitespace-only text means no semantic search. */
export const toGallerySemanticTextReference = (text: string): GallerySemanticReference | null => {
  const query = text.trim();

  return query ? { kind: 'text', query } : null;
};

/** The label-free shape a reference contributes to query cache keys. */
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

/*
 * Persist only a registry key for dropped blobs. New files evict the previous entry; unresolved keys clear the
 * search.
 */

let nextId = 0;

const files = new Map<string, { blob: Blob; label: string }>();

export const registerExternalImageFile = (blob: Blob, label: string): string => {
  nextId += 1;
  // The counter alone is only unique per JS realm, while the id is persisted
  // in widget values; the random token keeps ids from two realms (another
  // tab, a reload) from colliding into the wrong blob.
  const fileId = `external-${String(nextId)}-${Math.random().toString(36).slice(2, 10)}`;

  files.clear();
  files.set(fileId, { blob, label });

  return fileId;
};

export const getExternalImageFile = (fileId: string): { blob: Blob; label: string } | null => files.get(fileId) ?? null;

/*
 * Keep large cluster membership lists in memory under one persisted key. Members include media kinds; new
 * registrations evict old ones and unresolved keys clear search.
 */

let nextClusterId = 0;

const clusters = new Map<string, { itemKeys: GalleryItemKey[]; label: string }>();

export const registerImageCluster = (itemKeys: GalleryItemKey[], label: string): string => {
  nextClusterId += 1;
  // Per-realm random tokens prevent persisted keys from resolving to unrelated clusters after reload or in another
  // tab.
  const clusterId = `cluster-${String(nextClusterId)}-${Math.random().toString(36).slice(2, 10)}`;

  clusters.clear();
  clusters.set(clusterId, { itemKeys, label });

  return clusterId;
};

export const getImageCluster = (clusterId: string): { itemKeys: GalleryItemKey[]; label: string } | null =>
  clusters.get(clusterId) ?? null;

/**
 * Prune deleted members from the client-owned cluster. Rollback restores only this call's removals and only while
 * the same registration owns the slot.
 */
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

    // Restore only this call's removals in their original order, preserving concurrent prunes and rollbacks.
    const restored = new Set([...current.itemKeys, ...removedKeys]);

    clusters.set(clusterId, {
      itemKeys: originalKeys.filter((key) => restored.has(key)),
      label: current.label,
    });
  };
};

/**
 * Parses a persisted widget value into a semantic reference. Tolerates the
 * legacy bare-image-name shape, and reads a file key that no longer resolves
 * in the registry as no search at all.
 */
/**
 * File and cluster references resolve only in their originating realm; positions within their rankings cannot
 * survive reload or cross-tab restoration.
 */
const isSessionScopedGallerySemanticReference = (value: unknown): boolean =>
  !!value &&
  typeof value === 'object' &&
  ((value as Record<string, unknown>).kind === 'file' || (value as Record<string, unknown>).kind === 'cluster');

/**
 * Drop existing positions tied to rejected rankings so they cannot become unrelated board-page offsets. Do not
 * create absent state.
 */
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

/**
 * Drop infinite-window anchors on restoration to avoid stranding users mid-board without pagination controls.
 * Preserve actual paginated page numbers.
 */
export const stripInfiniteWindowAnchor = (values: Record<string, unknown>): Record<string, unknown> | null =>
  values.paginationMode !== 'paginated' && typeof values.galleryPage === 'number' && values.galleryPage > 0
    ? { ...values, galleryPage: 0 }
    : null;

/**
 * Saving strips all realm-scoped references, even those that currently resolve, because another realm cannot use
 * them.
 */
export const stripSessionScopedGallerySearch = (values: Record<string, unknown>): Record<string, unknown> | null =>
  stripGallerySearchPositions(
    values,
    (value) =>
      isSessionScopedGallerySemanticReference(value) ||
      (value !== null && value !== undefined && parseGallerySemanticReference(value) === null)
  );

/** Adoption strips only unresolved references: same-session reopen, fork, or import can retain a live ranking. */
export const stripUnresolvableGallerySearch = (values: Record<string, unknown>): Record<string, unknown> | null =>
  stripGallerySearchPositions(
    values,
    (value) => value !== null && value !== undefined && parseGallerySemanticReference(value) === null
  );

export const parseGallerySemanticReference = (value: unknown): GallerySemanticReference | null => {
  // Legacy shape: a bare image name.
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

/**
 * Files take precedence over URLs. Resolve this app's image URLs by name to avoid private-address download
 * rejection; other HTTP(S) URLs use web-image search.
 */
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

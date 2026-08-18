/**
 * Identity of a semantic gallery search: a text prompt, a gallery image, a
 * web image URL, or a dropped file held in the external-image registry.
 */
export type GallerySemanticQuery =
  | { kind: 'text'; query: string }
  | { kind: 'image'; imageName: string }
  | { kind: 'url'; url: string }
  | { kind: 'file'; fileId: string };

/** A persisted semantic search: a text prompt, a gallery image, a web URL, or a dropped file. */
export type GallerySemanticReference =
  | { kind: 'text'; query: string }
  | { kind: 'image'; imageName: string }
  | { kind: 'url'; url: string }
  | { kind: 'file'; fileId: string; label: string };

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
  }
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
  }
};

/*
 * In-memory registry for externally dropped reference images (OS file drops).
 * Blobs cannot live in persisted widget values, so the persisted value keeps a
 * registry key; a key that no longer resolves (e.g. after a reload) reads as a
 * cleared search. Only one external query is active at a time, so registering
 * a new file evicts the previous one.
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

/**
 * Parses a persisted widget value into a semantic reference. Tolerates the
 * legacy bare-image-name shape, and reads a file key that no longer resolves
 * in the registry as no search at all.
 */
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
      // Dropped files live in an in-memory registry; a persisted key that no
      // longer resolves (e.g. after a reload) reads as no search at all.
      if (getExternalImageFile(record.fileId) === null) {
        return null;
      }

      return {
        fileId: record.fileId,
        kind: 'file',
        label: typeof record.label === 'string' && record.label ? record.label : 'dropped image',
      };
    }
  }

  return null;
};

/** Matches this app's own image URLs so in-app drags become by-name queries. */
const APP_IMAGE_PATH = /\/api\/v\d+\/images\/i\/([^/]+)\//;

/**
 * Interpret a native drop for image-similarity search. Files win over URLs;
 * a URL pointing at this app's own image endpoint becomes a by-name query
 * (dragging the preview image in must not round-trip through the server's
 * URL downloader — it would be refused as a private-address fetch); anything
 * else http(s) is searched as a web image. Returns null for unusable drops.
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

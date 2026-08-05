import { apiFetch, apiFetchJson, apiFetchRaw } from '@platform/transport/http';

/**
 * The image half of a project file: pulling referenced bytes off the server on
 * the way out, and putting the missing ones back on the way in.
 *
 * This lives beside the archive rather than reaching for
 * `canvas-operations/backend/canvasImages.ts`, which does the same upload for
 * paint layers. That module is private to canvas-operations, and exporting it
 * would make importing a project file — a Launchpad action, on a route that
 * deliberately never loads the editor — pull the canvas graph behind it.
 *
 * Restored images are uploaded with `image_category: 'other'`. That is the
 * canvas's private category: the backend lists it in neither `IMAGE_CATEGORIES`
 * nor `ASSETS_CATEGORIES`, so it appears in no gallery view and no board count
 * (the reasoning is written out in full at `canvas-operations/backend/canvasImages.ts`).
 * The previous frontend uploaded restored images as `'general'`, which empties
 * a stranger's project into your gallery; the pixels a document points at are
 * not gallery content, they are the document.
 */

const IMAGES_BASE = '/api/v1/images';

/**
 * Names per existence request. The endpoint answers any length, but a URL-free
 * POST body of ten thousand names is still a request worth splitting so that one
 * slow lookup cannot stall the whole check.
 */
const EXISTENCE_BATCH_SIZE = 500;

export interface UploadedImage {
  height: number;
  imageName: string;
  width: number;
}

interface ImageDTOSubset {
  height: number;
  image_name: string;
  width: number;
}

/**
 * Which of `imageNames` the server already has. The endpoint silently omits
 * names it cannot serve, which is exactly the answer wanted here: anything
 * missing from the response has to come out of the archive.
 */
export const findExistingImageNames = async (
  imageNames: readonly string[],
  signal?: AbortSignal
): Promise<Set<string>> => {
  const existing = new Set<string>();

  for (let offset = 0; offset < imageNames.length; offset += EXISTENCE_BATCH_SIZE) {
    const batch = imageNames.slice(offset, offset + EXISTENCE_BATCH_SIZE);
    const found = await apiFetchJson<ImageDTOSubset[]>(`${IMAGES_BASE}/images_by_names`, {
      body: JSON.stringify({ image_names: batch }),
      method: 'POST',
      signal,
    });

    for (const dto of found) {
      existing.add(dto.image_name);
    }
  }

  return existing;
};

/**
 * Full-resolution bytes for one image, or `null` when the server will not serve
 * it. A missing image is not an export failure: the previous frontend logged and
 * skipped, and a project that exports every layer but one is far more use than
 * one that refuses to export at all.
 */
export const fetchImageBytes = async (imageName: string, signal?: AbortSignal): Promise<Uint8Array | null> => {
  const response = await apiFetchRaw(`${IMAGES_BASE}/i/${encodeURIComponent(imageName)}/full`, { signal });

  if (!response.ok) {
    return null;
  }

  return new Uint8Array(await response.arrayBuffer());
};

export interface FetchedThumbnail {
  bytes: Uint8Array;
  /** Response MIME type, which decides the cover entry's extension. */
  contentType: string;
}

/** Thumbnail bytes for the cover entry. `null` when the server will not serve it. */
export const fetchImageThumbnail = async (
  imageName: string,
  signal?: AbortSignal
): Promise<FetchedThumbnail | null> => {
  const response = await apiFetchRaw(`${IMAGES_BASE}/i/${encodeURIComponent(imageName)}/thumbnail`, { signal });

  if (!response.ok) {
    return null;
  }

  return {
    bytes: new Uint8Array(await response.arrayBuffer()),
    contentType: response.headers.get('content-type') ?? 'image/webp',
  };
};

/**
 * Put archived bytes back on the server. The returned name is authoritative and
 * frequently differs from the archived one — the server names images itself —
 * which is why every import ends with a remapping pass.
 */
export const uploadArchiveImage = async (
  bytes: Uint8Array,
  fileName: string,
  options: { contentType?: string; signal?: AbortSignal } = {}
): Promise<UploadedImage> => {
  const query = new URLSearchParams({ image_category: 'other', is_intermediate: 'false' });
  const body = new FormData();

  body.append(
    'file',
    new File([bytes as BlobPart], fileName, { type: options.contentType || 'application/octet-stream' })
  );

  const response = await apiFetch(`${IMAGES_BASE}/upload?${query.toString()}`, {
    body,
    method: 'POST',
    signal: options.signal,
  });
  const dto = (await response.json()) as ImageDTOSubset;

  return { height: dto.height, imageName: dto.image_name, width: dto.width };
};

const EXTENSION_BY_MIME: Readonly<Record<string, string>> = {
  'image/jpeg': 'jpg',
  'image/png': 'png',
  'image/webp': 'webp',
};

/** Extension for a cover entry, given what the server said it served. */
export const coverExtensionForMime = (contentType: string): string =>
  EXTENSION_BY_MIME[contentType.split(';')[0]!.trim().toLowerCase()] ?? 'png';

const MIME_BY_EXTENSION: Readonly<Record<string, string>> = {
  jpeg: 'image/jpeg',
  jpg: 'image/jpeg',
  png: 'image/png',
  webp: 'image/webp',
};

/** Best guess at a bundled entry's MIME type, from its name. */
export const mimeForEntryName = (entryName: string): string =>
  MIME_BY_EXTENSION[entryName.split('.').pop()?.toLowerCase() ?? ''] ?? 'image/png';

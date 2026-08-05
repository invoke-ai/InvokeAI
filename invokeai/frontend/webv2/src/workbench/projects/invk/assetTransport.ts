import { mapWithConcurrency } from '@platform/core/concurrency';
import { apiFetch, apiFetchJson, apiFetchRaw, HttpRequestIdentityExpiredError } from '@platform/transport/http';

/**
 * The asset half of a project file: pulling referenced bytes off the server on
 * the way out, and putting the missing ones back on the way in.
 *
 * This lives beside the archive rather than reaching for
 * `canvas-operations/backend/canvasImages.ts`, which does the same upload for
 * paint layers. That module is private to canvas-operations, and exporting it
 * would make importing a project file — a Launchpad action, on a route that
 * deliberately never loads the editor — pull the canvas graph behind it.
 *
 * Restored assets are uploaded with the category `'other'`. That is the canvas's
 * private category: the backend lists it in neither `IMAGE_CATEGORIES` nor
 * `ASSETS_CATEGORIES`, so it appears in no gallery view and no board count (the
 * reasoning is written out in full at `canvas-operations/backend/canvasImages.ts`).
 * The previous frontend uploaded restored images as `'general'`, which empties a
 * stranger's project into your gallery; the pixels a document points at are not
 * gallery content, they are the document.
 *
 * ### Images and videos are not symmetric
 *
 * Everything here exists twice because the backend keeps the two in separate
 * namespaces with separate routes. The one place they genuinely differ in shape
 * is the existence check: images have a bulk `images_by_names`, videos have no
 * equivalent, so the video check fans out one request per name behind a
 * concurrency limit. That is what `hydrateVideoRefs` in the gallery's data layer
 * already does for the same reason, and what the previous frontend did for
 * images before the bulk endpoint existed.
 */

const IMAGES_BASE = '/api/v1/images';
const VIDEOS_BASE = '/api/v1/videos';

/**
 * Whether a failed request means "the work this belonged to is over" rather
 * than "this one asset could not be served".
 *
 * Export treats an unservable asset as a skip, which is right — half a
 * project's pixels beats none. But an aborted signal makes *every* asset
 * unservable, and a skip-everything export writes an archive full of nothing
 * and hands it to the browser as though it had succeeded. The two have to be
 * told apart at the point the request fails, because by the time the archive is
 * packed they look identical.
 */
export const isRequestCancellation = (error: unknown): boolean =>
  error instanceof HttpRequestIdentityExpiredError || (error instanceof Error && error.name === 'AbortError');

/**
 * Names per existence request. The endpoint answers any length, but a URL-free
 * POST body of ten thousand names is still a request worth splitting so that one
 * slow lookup cannot stall the whole check.
 */
const EXISTENCE_BATCH_SIZE = 500;

/** Simultaneous per-name video lookups, matching the image fetch limit. */
const VIDEO_EXISTENCE_CONCURRENCY = 5;

export interface UploadedImage {
  height: number;
  imageName: string;
  width: number;
}

export interface UploadedVideo {
  videoName: string;
}

interface ImageDTOSubset {
  height: number;
  image_name: string;
  width: number;
}

interface VideoDTOSubset {
  video_name: string;
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
 * Which of `videoNames` the server already has.
 *
 * There is no bulk equivalent of `images_by_names` for videos, so this asks per
 * name behind a concurrency limit. A 404 is the answer, not an error — the whole
 * question is which names are absent.
 */
export const findExistingVideoNames = async (
  videoNames: readonly string[],
  signal?: AbortSignal
): Promise<Set<string>> => {
  const found = await mapWithConcurrency(videoNames, VIDEO_EXISTENCE_CONCURRENCY, async (videoName) => {
    const response = await apiFetchRaw(`${VIDEOS_BASE}/i/${encodeURIComponent(videoName)}`, { signal });

    return response.ok ? videoName : null;
  });

  return new Set(found.filter((videoName): videoName is string => videoName !== null));
};

/**
 * Full-resolution bytes for one image, or `null` when the server will not serve
 * it. A missing asset is not an export failure: the previous frontend logged and
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

/** The same, for a video. */
export const fetchVideoBytes = async (videoName: string, signal?: AbortSignal): Promise<Uint8Array | null> => {
  const response = await apiFetchRaw(`${VIDEOS_BASE}/i/${encodeURIComponent(videoName)}/full`, { signal });

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

/** The same, for a video. Dimensions are not read: nothing in the document needs them. */
export const uploadArchiveVideo = async (
  bytes: Uint8Array,
  fileName: string,
  options: { contentType?: string; signal?: AbortSignal } = {}
): Promise<UploadedVideo> => {
  const query = new URLSearchParams({ is_intermediate: 'false', video_category: 'other' });
  const body = new FormData();

  body.append('file', new File([bytes as BlobPart], fileName, { type: options.contentType || 'video/mp4' }));

  const response = await apiFetch(`${VIDEOS_BASE}/upload?${query.toString()}`, {
    body,
    method: 'POST',
    signal: options.signal,
  });
  const dto = (await response.json()) as VideoDTOSubset;

  return { videoName: dto.video_name };
};

const EXTENSION_BY_MIME: Readonly<Record<string, string>> = {
  'image/jpeg': 'jpg',
  'image/png': 'png',
  'image/webp': 'webp',
};

/** Extension for a cover entry, given what the server said it served. */
export const coverExtensionForMime = (contentType: string): string =>
  EXTENSION_BY_MIME[contentType.split(';')[0]!.trim().toLowerCase()] ?? 'png';

const IMAGE_MIME_BY_EXTENSION: Readonly<Record<string, string>> = {
  jpeg: 'image/jpeg',
  jpg: 'image/jpeg',
  png: 'image/png',
  webp: 'image/webp',
};

/**
 * Only MP4 survives the backend's upload path — it probes the container and
 * 415s anything else (`_is_mp4_file` in `videos.py`) — so this is a table of
 * what a bundled entry might be *named*, not of what will be accepted.
 */
const VIDEO_MIME_BY_EXTENSION: Readonly<Record<string, string>> = {
  m4v: 'video/mp4',
  mkv: 'video/x-matroska',
  mov: 'video/quicktime',
  mp4: 'video/mp4',
  webm: 'video/webm',
};

/**
 * Best guess at a bundled entry's MIME type, from its name and the folder it
 * came out of. Only ever a hint for the upload's multipart part.
 *
 * The fallback is per kind because the receiving endpoint checks it. A video
 * whose extension is not in the table falling back to `image/png` would be
 * announced to `/videos/upload` as an image and refused before its bytes were
 * ever read — the entry's folder already established what it is, so the guess
 * should never contradict it.
 */
export const mimeForEntryName = (entryName: string, kind: 'image' | 'video' = 'image'): string => {
  const extension = entryName.split('.').pop()?.toLowerCase() ?? '';

  return kind === 'video'
    ? (VIDEO_MIME_BY_EXTENSION[extension] ?? 'video/mp4')
    : (IMAGE_MIME_BY_EXTENSION[extension] ?? 'image/png');
};

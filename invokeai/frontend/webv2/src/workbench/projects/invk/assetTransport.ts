import { mapWithConcurrency } from '@platform/core/concurrency';
import {
  apiFetch,
  apiFetchJson,
  apiFetchRaw,
  assertOk,
  HttpRequestIdentityExpiredError,
} from '@platform/transport/http';

import type { InvkMediaCategory } from './board';

import { INVK_MAX_ARCHIVE_BYTES } from './archive';
import { InvkFormatError } from './format';

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
 * ### Two upload paths, because a restore carries two kinds of media
 *
 * A document *reference* is a pointer to pixels the project draws with. Those go
 * up under the category `'other'` and onto no board — that is the canvas's
 * private category: the backend lists it in neither `IMAGE_CATEGORIES` nor
 * `ASSETS_CATEGORIES`, so it appears in no gallery view and no board count (the
 * reasoning is written out in full at `canvas-operations/backend/canvasImages.ts`).
 * The previous frontend uploaded restored images as `'general'`, which empties a
 * stranger's project into your gallery; the pixels a document points at are not
 * gallery content, they are the document.
 *
 * A project *board item* is the opposite: it is gallery content, and the archive
 * recorded the category it was filed under. {@link uploadBoardImage} and its
 * video twin put it back under that category, on the project's staging board, so
 * the restored board looks like the exported one rather than like a pile of
 * uncategorized uploads.
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

const BOARDS_BASE = '/api/v1/boards';
const IMAGES_BASE = '/api/v1/images';
const VIDEOS_BASE = '/api/v1/videos';

/** The backend truncates board names at 300 characters; doing it here keeps the name we chose. */
const MAX_BOARD_NAME_LENGTH = 300;

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

export type AssetResponseReader = (response: Response, signal?: AbortSignal) => Promise<Uint8Array>;

const createAssetResponseByteWriter = () => {
  const chunks: Uint8Array[] = [];
  let byteLength = 0;

  return {
    finish: () => {
      if (chunks.length === 0) {
        return new Uint8Array();
      }

      if (chunks.length === 1) {
        return chunks[0]!;
      }

      const bytes = new Uint8Array(byteLength);
      let offset = 0;

      for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
      }

      return bytes;
    },
    write: (chunk: Uint8Array) => {
      chunks.push(chunk);
      byteLength += chunk.byteLength;
    },
  };
};

const getDeclaredContentLength = (response: Response): number | null => {
  const value = response.headers.get('content-length')?.trim();

  if (value === undefined || !/^\d+$/.test(value)) {
    return null;
  }

  const byteLength = Number(value);

  return Number.isSafeInteger(byteLength) ? byteLength : Number.POSITIVE_INFINITY;
};

/**
 * A single export-scoped reader. Every response is streamed through the same
 * fixed byte budget, so both one hostile response and many individually-small
 * responses stop before they can materialize an oversized project in memory.
 */
export const createAssetResponseReader = (): AssetResponseReader => {
  type AssetStreamReader = ReadableStreamDefaultReader<Uint8Array>;

  const activeReaders = new Set<AssetStreamReader>();
  const cancellationRequests = new WeakMap<AssetStreamReader, Promise<void>>();
  let fatalRefusal: InvkFormatError | null = null;
  let usedBytes = 0;
  let reservedBytes = 0;

  const cancelReader = (reader: AssetStreamReader, reason: unknown): Promise<void> => {
    const existing = cancellationRequests.get(reader);

    if (existing !== undefined) {
      return existing;
    }

    const request = reader.cancel(reason).catch(() => undefined);

    cancellationRequests.set(reader, request);

    return request;
  };

  const refuseTooLarge = (message: string): InvkFormatError => {
    if (fatalRefusal === null) {
      fatalRefusal = new InvkFormatError('too-large', message);

      for (const reader of activeReaders) {
        void cancelReader(reader, fatalRefusal);
      }
    }

    return fatalRefusal;
  };

  return async (response, signal) => {
    const body = response.body;
    const declaredByteLength = getDeclaredContentLength(response);

    const refuseBeforeReading = async (error: unknown): Promise<never> => {
      try {
        await body?.cancel(error);
      } catch {
        // The refusal is authoritative even if the transport cannot be cancelled.
      }

      throw error;
    };

    if (fatalRefusal !== null) {
      return refuseBeforeReading(fatalRefusal);
    }

    if (signal?.aborted) {
      return refuseBeforeReading(signal.reason);
    }

    if (declaredByteLength !== null && declaredByteLength > INVK_MAX_ARCHIVE_BYTES) {
      return refuseBeforeReading(refuseTooLarge(`Asset response is larger than ${INVK_MAX_ARCHIVE_BYTES} bytes.`));
    }

    if (declaredByteLength !== null && usedBytes + reservedBytes + declaredByteLength > INVK_MAX_ARCHIVE_BYTES) {
      return refuseBeforeReading(refuseTooLarge(`Project assets exceed ${INVK_MAX_ARCHIVE_BYTES} bytes.`));
    }

    const writer = createAssetResponseByteWriter();
    let remainingReservation = declaredByteLength ?? 0;
    let responseBytes = 0;

    reservedBytes += remainingReservation;

    if (body === null) {
      reservedBytes -= remainingReservation;

      return writer.finish();
    }

    const reader = body.getReader();

    activeReaders.add(reader);

    let finished = false;
    const cancelOnAbort = () => {
      void cancelReader(reader, signal?.reason);
    };

    signal?.addEventListener('abort', cancelOnAbort, { once: true });

    try {
      signal?.throwIfAborted();

      while (true) {
        const result = await reader.read();

        if (fatalRefusal !== null) {
          throw fatalRefusal;
        }

        signal?.throwIfAborted();

        if (result.done) {
          finished = true;
          reservedBytes -= remainingReservation;
          remainingReservation = 0;

          return writer.finish();
        }

        const chunkBytes = result.value.byteLength;
        const reservationConsumed = Math.min(remainingReservation, chunkBytes);
        const bytesPastDeclaration = chunkBytes - reservationConsumed;

        if (responseBytes + chunkBytes > INVK_MAX_ARCHIVE_BYTES) {
          throw refuseTooLarge(`Asset response exceeds ${INVK_MAX_ARCHIVE_BYTES} bytes.`);
        }

        if (usedBytes + reservedBytes + bytesPastDeclaration > INVK_MAX_ARCHIVE_BYTES) {
          throw refuseTooLarge(`Project assets exceed ${INVK_MAX_ARCHIVE_BYTES} bytes.`);
        }

        responseBytes += chunkBytes;
        usedBytes += chunkBytes;
        reservedBytes -= reservationConsumed;
        remainingReservation -= reservationConsumed;
        writer.write(result.value);
      }
    } catch (error) {
      usedBytes -= responseBytes;
      reservedBytes -= remainingReservation;
      responseBytes = 0;
      remainingReservation = 0;

      if (!finished) {
        await cancelReader(reader, error);
      }

      throw fatalRefusal ?? error;
    } finally {
      activeReaders.delete(reader);
      signal?.removeEventListener('abort', cancelOnAbort);
      reader.releaseLock();
    }
  };
};

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

    if (response.ok) {
      return videoName;
    }

    if (response.status === 403 || response.status === 404) {
      return null;
    }

    await assertOk(response);
    return null;
  });

  return new Set(found.filter((videoName): videoName is string => videoName !== null));
};

/**
 * Full-resolution bytes for one image, or `null` when the server will not serve
 * it. A missing asset is not an export failure: the previous frontend logged and
 * skipped, and a project that exports every layer but one is far more use than
 * one that refuses to export at all.
 */
const fetchImageBytesWithReader = async (
  imageName: string,
  signal: AbortSignal | undefined,
  readResponse: AssetResponseReader
): Promise<Uint8Array | null> => {
  const response = await apiFetchRaw(`${IMAGES_BASE}/i/${encodeURIComponent(imageName)}/full`, { signal });

  if (!response.ok) {
    return null;
  }

  return readResponse(response, signal);
};

export const fetchImageBytes = (imageName: string, signal?: AbortSignal): Promise<Uint8Array | null> =>
  fetchImageBytesWithReader(imageName, signal, createAssetResponseReader());

/** The same, for a video. */
const fetchVideoBytesWithReader = async (
  videoName: string,
  signal: AbortSignal | undefined,
  readResponse: AssetResponseReader
): Promise<Uint8Array | null> => {
  const response = await apiFetchRaw(`${VIDEOS_BASE}/i/${encodeURIComponent(videoName)}/full`, { signal });

  if (!response.ok) {
    return null;
  }

  return readResponse(response, signal);
};

export const fetchVideoBytes = (videoName: string, signal?: AbortSignal): Promise<Uint8Array | null> =>
  fetchVideoBytesWithReader(videoName, signal, createAssetResponseReader());

export interface FetchedThumbnail {
  bytes: Uint8Array;
  /** Response MIME type, which decides the cover entry's extension. */
  contentType: string;
}

/** Thumbnail bytes for the cover entry. `null` when the server will not serve it. */
const fetchImageThumbnailWithReader = async (
  imageName: string,
  signal: AbortSignal | undefined,
  readResponse: AssetResponseReader
): Promise<FetchedThumbnail | null> => {
  const response = await apiFetchRaw(`${IMAGES_BASE}/i/${encodeURIComponent(imageName)}/thumbnail`, { signal });

  if (!response.ok) {
    return null;
  }

  return {
    bytes: await readResponse(response, signal),
    contentType: response.headers.get('content-type') ?? 'image/webp',
  };
};

export const fetchImageThumbnail = (imageName: string, signal?: AbortSignal): Promise<FetchedThumbnail | null> =>
  fetchImageThumbnailWithReader(imageName, signal, createAssetResponseReader());

/** Production export transport with one cumulative response budget. */
export const createAssetExportTransport = () => {
  const readResponse = createAssetResponseReader();

  return {
    fetchImageBytes: (imageName: string, signal?: AbortSignal) =>
      fetchImageBytesWithReader(imageName, signal, readResponse),
    fetchImageThumbnail: (imageName: string, signal?: AbortSignal) =>
      fetchImageThumbnailWithReader(imageName, signal, readResponse),
    fetchVideoBytes: (videoName: string, signal?: AbortSignal) =>
      fetchVideoBytesWithReader(videoName, signal, readResponse),
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

export interface BoardUploadOptions {
  /** The project's staging board. Board media is never uploaded unboarded. */
  boardId: string;
  /** The category the exporting server had it filed under. */
  category: InvkMediaCategory;
  contentType?: string;
  signal?: AbortSignal;
}

/**
 * Put one of the project board's images back, under its archived category and on
 * the staging board the new project is about to claim.
 *
 * The returned name is a genuinely new identity, always — `board_images` has
 * `PRIMARY KEY (image_name)`, so an image sits on exactly one board and a restore
 * that reused an existing name would be moving a stranger's picture onto this
 * project's board rather than copying it.
 */
export const uploadBoardImage = async (
  bytes: Uint8Array,
  fileName: string,
  options: BoardUploadOptions
): Promise<UploadedImage> => {
  const query = new URLSearchParams({
    board_id: options.boardId,
    image_category: options.category,
    is_intermediate: 'false',
  });
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

/** The same, for one of the board's videos. */
export const uploadBoardVideo = async (
  bytes: Uint8Array,
  fileName: string,
  options: BoardUploadOptions
): Promise<UploadedVideo> => {
  const query = new URLSearchParams({
    board_id: options.boardId,
    is_intermediate: 'false',
    video_category: options.category,
  });
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

/** One copy the server made, under the identity it assigned. */
export interface CopiedMedia {
  name: string;
  sourceName: string;
}

export interface CopyMediaResult {
  /** Source names the server would not copy. Reported per name, never fatal for the batch. */
  failed: string[];
  copied: CopiedMedia[];
}

/**
 * Copy media onto a board without the bytes leaving the server.
 *
 * Duplication needs new identities for the same reason import does — `board_images` keys on the
 * image name — but both projects live on this one server, so round-tripping the pixels through the
 * browser would cost twice the board's size in traffic, two requests per item, and would run into
 * the same cumulative budget that bounds an export. The endpoint carries category, origin and the
 * metadata embedded in the PNG; starring is not part of a copy, so callers star afterwards through
 * the same path an import uses.
 */
export const copyImagesToBoard = async (
  imageNames: readonly string[],
  boardId: string,
  signal?: AbortSignal
): Promise<CopyMediaResult> => {
  if (imageNames.length === 0) {
    return { copied: [], failed: [] };
  }

  const body = await apiFetchJson<{
    copied?: { image_name: string; source_image_name: string }[];
    failed?: string[];
  }>(`${IMAGES_BASE}/copy`, {
    body: JSON.stringify({ board_id: boardId, image_names: imageNames }),
    method: 'POST',
    signal,
  });

  return {
    copied: (body.copied ?? []).map((entry) => ({ name: entry.image_name, sourceName: entry.source_image_name })),
    failed: body.failed ?? [],
  };
};

/** The same, for videos. */
export const copyVideosToBoard = async (
  videoNames: readonly string[],
  boardId: string,
  signal?: AbortSignal
): Promise<CopyMediaResult> => {
  if (videoNames.length === 0) {
    return { copied: [], failed: [] };
  }

  const body = await apiFetchJson<{
    copied?: { source_video_name: string; video_name: string }[];
    failed?: string[];
  }>(`${VIDEOS_BASE}/copy`, {
    body: JSON.stringify({ board_id: boardId, video_names: videoNames }),
    method: 'POST',
    signal,
  });

  return {
    copied: (body.copied ?? []).map((entry) => ({ name: entry.video_name, sourceName: entry.source_video_name })),
    failed: body.failed ?? [],
  };
};

/** Names the server would not star, out of the names asked for. */
export interface BulkStarResult {
  failed: string[];
}

/**
 * What a bulk star actually achieved.
 *
 * Both endpoints answer with the names they *did* star, and only the video one
 * also lists failures — a name it skipped (a foreign one, or one deleted between
 * the upload and the star) appears in neither list. Deriving the failures from
 * the success list rather than reading `failed_*` therefore covers both routes
 * with one rule, and the rule is the honest one: anything not reported as starred
 * did not get starred.
 */
const toBulkStarResult = (requested: readonly string[], starred: readonly string[]): BulkStarResult => {
  const succeeded = new Set(starred);

  return { failed: requested.filter((name) => !succeeded.has(name)) };
};

/** Star restored images. Starring is never part of an upload or a copy — it is always this call. */
export const starImages = async (imageNames: readonly string[], signal?: AbortSignal): Promise<BulkStarResult> => {
  if (imageNames.length === 0) {
    return { failed: [] };
  }

  const body = await apiFetchJson<{ starred_images?: string[] }>(`${IMAGES_BASE}/star`, {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
    signal,
  });

  return toBulkStarResult(imageNames, body.starred_images ?? []);
};

/** The same, for videos. */
export const starVideos = async (videoNames: readonly string[], signal?: AbortSignal): Promise<BulkStarResult> => {
  if (videoNames.length === 0) {
    return { failed: [] };
  }

  const body = await apiFetchJson<{ starred_videos?: string[] }>(`${VIDEOS_BASE}/star`, {
    body: JSON.stringify({ video_names: videoNames }),
    method: 'POST',
    signal,
  });

  return toBulkStarResult(videoNames, body.starred_videos ?? []);
};

/**
 * An unclaimed private board for a restore to upload into, which project creation
 * then claims and renames.
 *
 * Creating the project first would mean posting it with the old asset names and
 * then updating it with the remapped ones, making the *update* the commit point:
 * a failure there leaves a real project full of references to media that never
 * arrived. Staging inverts that — the media is in place before the project
 * exists, and the create either claims the board and its contents or leaves
 * nothing a person can see.
 */
export const createStagingBoard = async (boardName: string, signal?: AbortSignal): Promise<string> => {
  const query = new URLSearchParams({ board_name: boardName.slice(0, MAX_BOARD_NAME_LENGTH) });
  const dto = await apiFetchJson<{ board_id: string }>(`${BOARDS_BASE}/?${query.toString()}`, {
    method: 'POST',
    signal,
  });

  return dto.board_id;
};

/**
 * Drop a staging board whose project was never created.
 *
 * `include_images=false` deliberately: the restore deletes the identities it
 * created itself, one by one, and anything else that wandered onto the board in
 * the meantime — a generation that finished while the import ran — must survive
 * as Uncategorized rather than be collected by a cleanup that never meant it.
 */
export const deleteStagingBoard = async (boardId: string, signal?: AbortSignal): Promise<void> => {
  await apiFetchJson<unknown>(`${BOARDS_BASE}/${encodeURIComponent(boardId)}?include_images=false`, {
    method: 'DELETE',
    signal,
  });
};

/** Delete image identities created by a restore whose project could not be created. */
export const deleteArchiveImages = async (imageNames: string[], signal?: AbortSignal): Promise<void> => {
  if (imageNames.length === 0) {
    return;
  }

  await apiFetchJson<unknown>(`${IMAGES_BASE}/delete`, {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
    signal,
  });
};

/** Delete video identities created by a restore whose project could not be created. */
export const deleteArchiveVideos = async (videoNames: string[], signal?: AbortSignal): Promise<void> => {
  if (videoNames.length === 0) {
    return;
  }

  await apiFetchJson<unknown>(`${VIDEOS_BASE}/delete`, {
    body: JSON.stringify({ video_names: videoNames }),
    method: 'POST',
    signal,
  });
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

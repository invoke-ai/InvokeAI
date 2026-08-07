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
 * The asset half of a project file: pulling referenced bytes off the server on the way out, and
 * putting the missing ones back on the way in.
 *
 * Not reached through `canvas-operations/backend/canvasImages.ts`, which does the same upload:
 * importing is a Launchpad action on a route that never loads the editor, and it must not pull the
 * canvas graph behind it.
 *
 * Document references upload under category `'other'` — the canvas's private category, in neither
 * `IMAGE_CATEGORIES` nor `ASSETS_CATEGORIES` — because pixels a document points at are the
 * document, not gallery content. Board items upload under the category the archive recorded.
 */

/** One budget for the whole transfer: a per-phase limit would multiply what it exists to bound. */
export const INVK_TRANSFER_CONCURRENCY = 5;

const BOARDS_BASE = '/api/v1/boards';
const IMAGES_BASE = '/api/v1/images';
const VIDEOS_BASE = '/api/v1/videos';

/** The backend truncates board names at 300 characters; doing it here keeps the name we chose. */
const MAX_BOARD_NAME_LENGTH = 300;

/**
 * Whether a failure means "this work is over" rather than "this one asset could not be served".
 *
 * An aborted signal makes *every* asset unservable, and a skip-everything export writes an empty
 * archive and hands it over as a success. Distinguishable only where the request fails — by the
 * time the archive is packed the two look identical.
 */
export const isRequestCancellation = (error: unknown): boolean =>
  error instanceof HttpRequestIdentityExpiredError || (error instanceof Error && error.name === 'AbortError');

/** Names per existence request, so one slow lookup cannot stall the whole check. */
const EXISTENCE_BATCH_SIZE = 500;

/** Maximum names the backend's synchronous batch routes accept. Adapters below own the split. */
const MEDIA_REQUEST_BATCH_SIZE = 1_000;

const toBatches = <T>(items: readonly T[], size: number): T[][] => {
  const batches: T[][] = [];

  for (let offset = 0; offset < items.length; offset += size) {
    batches.push(items.slice(offset, offset + size));
  }

  return batches;
};

/** `apiFetchRaw` returns the response untouched, so an unread body pins its connection. */
const discardBody = async (response: Response): Promise<void> => {
  try {
    await response.body?.cancel();
  } catch {
    // A body that cannot be cancelled is already finished with.
  }
};

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

const concatChunks = (chunks: readonly Uint8Array[], byteLength: number): Uint8Array => {
  if (chunks.length < 2) {
    return chunks[0] ?? new Uint8Array();
  }

  const bytes = new Uint8Array(byteLength);
  let offset = 0;

  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }

  return bytes;
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
 * A single export-scoped reader: every response streams through one fixed byte budget, so neither a
 * hostile response nor many small ones can materialize an oversized project in memory.
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

    const chunks: Uint8Array[] = [];
    let chunkBytesTotal = 0;
    let remainingReservation = declaredByteLength ?? 0;
    let responseBytes = 0;

    reservedBytes += remainingReservation;

    if (body === null) {
      reservedBytes -= remainingReservation;

      return concatChunks(chunks, chunkBytesTotal);
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

          return concatChunks(chunks, chunkBytesTotal);
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
        chunks.push(result.value);
        chunkBytesTotal += chunkBytes;
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

/** Which of `imageNames` the server already has; anything omitted must come out of the archive. */
export const findExistingImageNames = async (
  imageNames: readonly string[],
  signal?: AbortSignal
): Promise<Set<string>> => {
  // Through the same pool as everything else. Splitting the names was meant to stop one slow
  // lookup stalling the check, which a sequential loop gives straight back.
  const found = await mapWithConcurrency(
    toBatches(imageNames, EXISTENCE_BATCH_SIZE),
    INVK_TRANSFER_CONCURRENCY,
    (batch) =>
      apiFetchJson<ImageDTOSubset[]>(`${IMAGES_BASE}/images_by_names`, {
        body: JSON.stringify({ image_names: batch }),
        method: 'POST',
        signal,
      }),
    { signal }
  );

  return new Set(found.flat().map((dto) => dto.image_name));
};

/**
 * Which of `videoNames` the server already has. No bulk equivalent of `images_by_names` exists for
 * videos, so this asks per name; a 404 is the answer, not an error.
 */
export const findExistingVideoNames = async (
  videoNames: readonly string[],
  signal?: AbortSignal
): Promise<Set<string>> => {
  const found = await mapWithConcurrency(
    videoNames,
    INVK_TRANSFER_CONCURRENCY,
    async (videoName) => {
      const response = await apiFetchRaw(`${VIDEOS_BASE}/i/${encodeURIComponent(videoName)}`, { signal });

      if (response.ok) {
        // The DTO is not wanted, only its existence — but an unread body holds its connection open
        // until it is collected, and this runs once per referenced video.
        await discardBody(response);
        return videoName;
      }

      if (response.status === 403 || response.status === 404) {
        await discardBody(response);
        return null;
      }

      await assertOk(response);
      return null;
    },
    { signal }
  );

  return new Set(found.filter((videoName): videoName is string => videoName !== null));
};

/**
 * Bytes for one asset, or `null` when the server will not serve it. A missing asset is not an export
 * failure: a project that exports every layer but one is far more use than one that refuses.
 */
const fetchAsset = async (
  url: string,
  signal: AbortSignal | undefined,
  readResponse: AssetResponseReader
): Promise<{ bytes: Uint8Array; contentType: string | null } | null> => {
  const response = await apiFetchRaw(url, { signal });

  if (!response.ok) {
    await discardBody(response);
    return null;
  }

  return { bytes: await readResponse(response, signal), contentType: response.headers.get('content-type') };
};

const assetUrl = (base: string, name: string, variant: 'full' | 'thumbnail'): string =>
  `${base}/i/${encodeURIComponent(name)}/${variant}`;

const fetchImageBytesWithReader = async (
  imageName: string,
  signal: AbortSignal | undefined,
  readResponse: AssetResponseReader
): Promise<Uint8Array | null> =>
  (await fetchAsset(assetUrl(IMAGES_BASE, imageName, 'full'), signal, readResponse))?.bytes ?? null;

export const fetchImageBytes = (imageName: string, signal?: AbortSignal): Promise<Uint8Array | null> =>
  fetchImageBytesWithReader(imageName, signal, createAssetResponseReader());

const fetchVideoBytesWithReader = async (
  videoName: string,
  signal: AbortSignal | undefined,
  readResponse: AssetResponseReader
): Promise<Uint8Array | null> =>
  (await fetchAsset(assetUrl(VIDEOS_BASE, videoName, 'full'), signal, readResponse))?.bytes ?? null;

export const fetchVideoBytes = (videoName: string, signal?: AbortSignal): Promise<Uint8Array | null> =>
  fetchVideoBytesWithReader(videoName, signal, createAssetResponseReader());

export interface FetchedThumbnail {
  bytes: Uint8Array;
  /** Response MIME type, which decides the cover entry's extension. */
  contentType: string;
}

const fetchImageThumbnailWithReader = async (
  imageName: string,
  signal: AbortSignal | undefined,
  readResponse: AssetResponseReader
): Promise<FetchedThumbnail | null> => {
  const fetched = await fetchAsset(assetUrl(IMAGES_BASE, imageName, 'thumbnail'), signal, readResponse);

  return fetched === null ? null : { bytes: fetched.bytes, contentType: fetched.contentType ?? 'image/webp' };
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
 * Put bytes on the server. The returned name is authoritative and frequently differs from the one
 * asked for — the server names media itself — which is why every import ends with a remapping pass.
 *
 * A board upload's name is a genuinely new identity, always — see the `board_images` rule in
 * `transfer.ts`.
 */
const uploadMedia = async <T>(
  base: string,
  query: Record<string, string>,
  bytes: Uint8Array,
  fileName: string,
  defaultContentType: string,
  options: { contentType?: string; signal?: AbortSignal }
): Promise<T> => {
  const body = new FormData();

  body.append('file', new File([bytes as BlobPart], fileName, { type: options.contentType || defaultContentType }));

  const response = await apiFetch(`${base}/upload?${new URLSearchParams(query).toString()}`, {
    body,
    method: 'POST',
    signal: options.signal,
  });

  return (await response.json()) as T;
};

const IMAGE_UPLOAD_MIME = 'application/octet-stream';
const VIDEO_UPLOAD_MIME = 'video/mp4';

/** Dimensions travel for images because the document needs them; nothing needs a video's. */
const toUploadedImage = (dto: ImageDTOSubset): UploadedImage => ({
  height: dto.height,
  imageName: dto.image_name,
  width: dto.width,
});

export const uploadArchiveImage = async (
  bytes: Uint8Array,
  fileName: string,
  options: { contentType?: string; signal?: AbortSignal } = {}
): Promise<UploadedImage> =>
  toUploadedImage(
    await uploadMedia<ImageDTOSubset>(
      IMAGES_BASE,
      { image_category: 'other', is_intermediate: 'false' },
      bytes,
      fileName,
      IMAGE_UPLOAD_MIME,
      options
    )
  );

export const uploadArchiveVideo = async (
  bytes: Uint8Array,
  fileName: string,
  options: { contentType?: string; signal?: AbortSignal } = {}
): Promise<UploadedVideo> => ({
  videoName: (
    await uploadMedia<VideoDTOSubset>(
      VIDEOS_BASE,
      { is_intermediate: 'false', video_category: 'other' },
      bytes,
      fileName,
      VIDEO_UPLOAD_MIME,
      options
    )
  ).video_name,
});

export interface BoardUploadOptions {
  /** The project's staging board. Board media is never uploaded unboarded. */
  boardId: string;
  /** The category the exporting server had it filed under. */
  category: InvkMediaCategory;
  contentType?: string;
  signal?: AbortSignal;
}

export const uploadBoardImage = async (
  bytes: Uint8Array,
  fileName: string,
  options: BoardUploadOptions
): Promise<UploadedImage> =>
  toUploadedImage(
    await uploadMedia<ImageDTOSubset>(
      IMAGES_BASE,
      { board_id: options.boardId, image_category: options.category, is_intermediate: 'false' },
      bytes,
      fileName,
      IMAGE_UPLOAD_MIME,
      options
    )
  );

export const uploadBoardVideo = async (
  bytes: Uint8Array,
  fileName: string,
  options: BoardUploadOptions
): Promise<UploadedVideo> => ({
  videoName: (
    await uploadMedia<VideoDTOSubset>(
      VIDEOS_BASE,
      { board_id: options.boardId, is_intermediate: 'false', video_category: options.category },
      bytes,
      fileName,
      VIDEO_UPLOAD_MIME,
      options
    )
  ).video_name,
});

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

interface CopyMediaRequest<Entry> {
  boardId: string;
  endpoint: string;
  names: readonly string[];
  requestKey: 'image_names' | 'video_names';
  signal?: AbortSignal;
  toCopiedMedia: (entry: Entry) => CopiedMedia;
}

/** Enforce the synchronous-route ceiling once for every server-side copy transport. */
const copyMediaToBoard = async <Entry>({
  boardId,
  endpoint,
  names,
  requestKey,
  signal,
  toCopiedMedia,
}: CopyMediaRequest<Entry>): Promise<CopyMediaResult> => {
  const result: CopyMediaResult = { copied: [], failed: [] };
  const batches = toBatches(names, MEDIA_REQUEST_BATCH_SIZE);

  for (let batchIndex = 0; batchIndex < batches.length; batchIndex += 1) {
    const batch = batches[batchIndex]!;
    if (signal?.aborted) {
      result.failed.push(...batches.slice(batchIndex).flat());
      break;
    }

    try {
      // Deliberately unaborted: the server works synchronously and may create identities after the
      // browser disconnects, and its response is the only way the ledger learns to remove them.
      // Cancellation is honoured between requests instead.
      const body = await apiFetchJson<{ copied?: Entry[]; failed?: string[] }>(endpoint, {
        body: JSON.stringify({ board_id: boardId, [requestKey]: batch }),
        method: 'POST',
      });

      result.copied.push(...(body.copied ?? []).map(toCopiedMedia));
      result.failed.push(...(body.failed ?? []));
    } catch (error) {
      if (isRequestCancellation(error)) {
        throw error;
      }
      result.failed.push(...batch);
    }

    if (signal?.aborted) {
      result.failed.push(...batches.slice(batchIndex + 1).flat());
      break;
    }
  }

  return result;
};

/**
 * Copy media onto a board without the bytes leaving the server. Carries category, origin and
 * embedded metadata; starring is not part of a copy, so callers star afterwards.
 */
export const copyImagesToBoard = (
  imageNames: readonly string[],
  boardId: string,
  signal?: AbortSignal
): Promise<CopyMediaResult> =>
  copyMediaToBoard<{ image_name: string; source_image_name: string }>({
    boardId,
    endpoint: `${IMAGES_BASE}/copy`,
    names: imageNames,
    requestKey: 'image_names',
    signal,
    toCopiedMedia: (entry) => ({ name: entry.image_name, sourceName: entry.source_image_name }),
  });

/** The same, for videos. */
export const copyVideosToBoard = (
  videoNames: readonly string[],
  boardId: string,
  signal?: AbortSignal
): Promise<CopyMediaResult> =>
  copyMediaToBoard<{ source_video_name: string; video_name: string }>({
    boardId,
    endpoint: `${VIDEOS_BASE}/copy`,
    names: videoNames,
    requestKey: 'video_names',
    signal,
    toCopiedMedia: (entry) => ({ name: entry.video_name, sourceName: entry.source_video_name }),
  });

/** Names the server would not star, out of the names asked for. */
export interface BulkStarResult {
  failed: string[];
}

/**
 * Failures are derived from the success list, not read from `failed_*`: only the video endpoint
 * reports failures, and a name it silently skipped appears in neither list.
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

  const failed: string[] = [];

  for (const batch of toBatches(videoNames, MEDIA_REQUEST_BATCH_SIZE)) {
    try {
      const body = await apiFetchJson<{ starred_videos?: string[] }>(`${VIDEOS_BASE}/star`, {
        body: JSON.stringify({ video_names: batch }),
        method: 'POST',
        signal,
      });

      failed.push(...toBulkStarResult(batch, body.starred_videos ?? []).failed);
    } catch (error) {
      if (isRequestCancellation(error)) {
        throw error;
      }
      failed.push(...batch);
    }
  }

  return { failed };
};

/**
 * An unclaimed private board for a restore to upload into, which project creation then claims.
 *
 * This is what makes the create the commit point: the media is in place before the project exists,
 * so the create either claims the board and its contents or leaves nothing a person can see.
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
 * Drop a staging board whose project was never created. `include_images=false` deliberately: the
 * restore deletes its own identities one by one, and a generation that landed on the board
 * meanwhile must survive as Uncategorized.
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

  let firstError: unknown = null;

  for (const batch of toBatches(videoNames, MEDIA_REQUEST_BATCH_SIZE)) {
    try {
      await apiFetchJson<unknown>(`${VIDEOS_BASE}/delete`, {
        body: JSON.stringify({ video_names: batch }),
        method: 'POST',
        signal,
      });
    } catch (error) {
      if (isRequestCancellation(error)) {
        throw error;
      }
      firstError ??= error;
    }
  }

  if (firstError !== null) {
    throw firstError instanceof Error
      ? firstError
      : new Error('One or more video cleanup requests failed.', { cause: firstError });
  }
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

/** What a bundled entry might be *named*; only MP4 survives the backend's `_is_mp4_file` probe. */
const VIDEO_MIME_BY_EXTENSION: Readonly<Record<string, string>> = {
  m4v: 'video/mp4',
  mkv: 'video/x-matroska',
  mov: 'video/quicktime',
  mp4: 'video/mp4',
  webm: 'video/webm',
};

/**
 * Best guess at a bundled entry's MIME type. The fallback is per kind because the endpoint checks
 * it: an unknown video extension falling back to `image/png` would be refused by `/videos/upload`.
 */
export const mimeForEntryName = (entryName: string, kind: 'image' | 'video' = 'image'): string => {
  const extension = entryName.split('.').pop()?.toLowerCase() ?? '';

  return kind === 'video'
    ? (VIDEO_MIME_BY_EXTENSION[extension] ?? 'video/mp4')
    : (IMAGE_MIME_BY_EXTENSION[extension] ?? 'image/png');
};

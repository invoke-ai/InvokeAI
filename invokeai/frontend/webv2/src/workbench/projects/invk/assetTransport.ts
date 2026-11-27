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
import { clampBoardName } from './board';
import { InvkFormatError } from './format';

/** One budget for the whole transfer: a per-phase limit would multiply what it exists to bound. */
export const INVK_TRANSFER_CONCURRENCY = 5;

const BOARDS_BASE = '/api/v1/boards';
const IMAGES_BASE = '/api/v1/images';
const VIDEOS_BASE = '/api/v1/videos';

/** Cancellation aborts the operation rather than reporting remaining assets as missing. */
export const isRequestCancellation = (error: unknown): boolean =>
  error instanceof HttpRequestIdentityExpiredError || (error instanceof Error && error.name === 'AbortError');

const EXISTENCE_BATCH_SIZE = 500;

/** The backend's synchronous batch route limit (`invokeai/app/api/routers/_limits.py`); adapters below split by it. */
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

export const findExistingImageNames = async (
  imageNames: readonly string[],
  signal?: AbortSignal
): Promise<Set<string>> => {
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

/** Videos need individual existence requests because no bulk endpoint exists; 404 means absent. */
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

/** Use server-returned names. Board uploads must always receive fresh identities. */
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
  boardId: string;
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

export interface CopiedMedia {
  name: string;
  sourceName: string;
}

export interface CopyMediaResult {
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
      // Deliberately unaborted: the server may create identities after a disconnect, and only its response tells the ledger.
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

/** Server-side copies preserve metadata, category, and origin; starring is separate. */
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

export interface BulkStarResult {
  failed: string[];
}

/** Derive failures from missing successes because endpoint failure lists are incomplete. */
const toBulkStarResult = (requested: readonly string[], starred: readonly string[]): BulkStarResult => {
  const succeeded = new Set(starred);

  return { failed: requested.filter((name) => !succeeded.has(name)) };
};

const starMedia = async (
  names: readonly string[],
  endpoint: string,
  requestKey: 'image_names' | 'video_names',
  responseKey: 'starred_images' | 'starred_videos',
  signal?: AbortSignal
): Promise<BulkStarResult> => {
  const failed: string[] = [];

  for (const batch of toBatches(names, MEDIA_REQUEST_BATCH_SIZE)) {
    try {
      const body = await apiFetchJson<Partial<Record<typeof responseKey, string[]>>>(endpoint, {
        body: JSON.stringify({ [requestKey]: batch }),
        method: 'POST',
        signal,
      });

      failed.push(...toBulkStarResult(batch, body[responseKey] ?? []).failed);
    } catch (error) {
      if (isRequestCancellation(error)) {
        throw error;
      }
      failed.push(...batch);
    }
  }

  return { failed };
};

export const starImages = (imageNames: readonly string[], signal?: AbortSignal): Promise<BulkStarResult> =>
  starMedia(imageNames, `${IMAGES_BASE}/star`, 'image_names', 'starred_images', signal);

export const starVideos = (videoNames: readonly string[], signal?: AbortSignal): Promise<BulkStarResult> =>
  starMedia(videoNames, `${VIDEOS_BASE}/star`, 'video_names', 'starred_videos', signal);

/** Atomically claiming the staging board is the project-create commit point. */
export const createStagingBoard = async (boardName: string, signal?: AbortSignal): Promise<string> => {
  const query = new URLSearchParams({ board_name: clampBoardName(boardName) });
  const dto = await apiFetchJson<{ board_id: string }>(`${BOARDS_BASE}/?${query.toString()}`, {
    method: 'POST',
    signal,
  });

  return dto.board_id;
};

/** Only after the project exists: a board is a member of a project the server already has. */
export const placeBoardInProject = async (
  boardId: string,
  projectId: string,
  archived: boolean,
  signal?: AbortSignal
): Promise<void> => {
  await apiFetchJson<unknown>(`${BOARDS_BASE}/${encodeURIComponent(boardId)}`, {
    body: JSON.stringify({ project_id: projectId, ...(archived ? { archived: true } : {}) }),
    method: 'PATCH',
    signal,
  });
};

export const getBoardProjectId = async (boardId: string, signal?: AbortSignal): Promise<string | null> =>
  (await apiFetchJson<{ project_id?: string | null }>(`${BOARDS_BASE}/${encodeURIComponent(boardId)}`, { signal }))
    .project_id ?? null;

/** `include_images=false`: a generation that landed on the board meanwhile must survive as Uncategorized. */
export const deleteStagingBoard = async (boardId: string, signal?: AbortSignal): Promise<void> => {
  await apiFetchJson<unknown>(`${BOARDS_BASE}/${encodeURIComponent(boardId)}?include_images=false`, {
    method: 'DELETE',
    signal,
  });
};

const deleteArchiveMedia = async (
  names: readonly string[],
  endpoint: string,
  requestKey: 'image_names' | 'video_names',
  signal?: AbortSignal
): Promise<void> => {
  let firstError: unknown = null;

  for (const batch of toBatches(names, MEDIA_REQUEST_BATCH_SIZE)) {
    try {
      await apiFetchJson<unknown>(endpoint, {
        body: JSON.stringify({ [requestKey]: batch }),
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
      : new Error('One or more media cleanup requests failed.', { cause: firstError });
  }
};

export const deleteArchiveImages = (imageNames: string[], signal?: AbortSignal): Promise<void> =>
  deleteArchiveMedia(imageNames, `${IMAGES_BASE}/delete`, 'image_names', signal);

export const deleteArchiveVideos = (videoNames: string[], signal?: AbortSignal): Promise<void> =>
  deleteArchiveMedia(videoNames, `${VIDEOS_BASE}/delete`, 'video_names', signal);

const EXTENSION_BY_MIME: Readonly<Record<string, string>> = {
  'image/jpeg': 'jpg',
  'image/png': 'png',
  'image/webp': 'webp',
};

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

export const mimeForEntryName = (entryName: string, kind: 'image' | 'video' = 'image'): string => {
  const extension = entryName.split('.').pop()?.toLowerCase() ?? '';

  return kind === 'video'
    ? (VIDEO_MIME_BY_EXTENSION[extension] ?? 'video/mp4')
    : (IMAGE_MIME_BY_EXTENSION[extension] ?? 'image/png');
};

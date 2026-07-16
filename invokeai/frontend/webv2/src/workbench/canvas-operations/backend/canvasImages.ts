/**
 * Uploads canvas paint bitmaps to the backend as persistent, non-gallery
 * images. Paint layers reference their pixels by `imageName` (never by URL or
 * inline data), so the persisted workbench document — which autosaves to
 * localStorage (~5 MB) — stays ref-only and the pixels live server-side.
 *
 * `image_category='other'` + `is_intermediate=false` keeps the bitmap durable
 * (not garbage-collected as an intermediate) while hiding it from the gallery's
 * general/asset views.
 *
 * The `fetch` seam is injectable so this runs in node tests without a DOM.
 * Auth + base-URL resolution mirror the shared HTTP client so uploads carry the
 * same bearer token as every other authenticated request. Zero React.
 */

import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';

import { buildApiUrl, getAuthToken } from '@workbench/backend/http';

export type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';

/** The subset of the backend `ImageDTO` this module reads. */
interface UploadImageResponseDTO {
  image_name: string;
  width: number;
  height: number;
}

/** Options for {@link uploadCanvasImage}. */
export interface UploadCanvasImageOptions {
  /** Overrides the default `'other'` category. */
  imageCategory?: 'other' | 'general' | 'control' | 'mask' | 'user';
  /** Overrides the default `false` (persistent, not garbage-collected). */
  isIntermediate?: boolean;
  /** Adds the image to a board, if given. */
  boardId?: string;
  /** File name sent in the multipart part (defaults to `canvas-paint.png`). */
  fileName?: string;
  /** Optional image metadata sent as JSON in the multipart body. */
  metadata?: Record<string, unknown>;
  /** Optional backend resize dimensions sent as JSON in the multipart body. */
  resizeTo?: { width: number; height: number };
  /** Injectable `fetch` implementation (defaults to the global). */
  fetch?: typeof globalThis.fetch;
  /** Cancels the multipart request when its owning operation is superseded. */
  signal?: AbortSignal;
}

/** Thrown when a canvas-image upload fails (non-2xx or network error). */
export class CanvasImageUploadError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null) {
    super(message);
    this.name = 'CanvasImageUploadError';
    this.status = status;
  }
}

/**
 * POSTs `blob` as a PNG to `/api/v1/images/upload` (multipart `file` field) and
 * returns the server-assigned image name and dimensions. Throws
 * {@link CanvasImageUploadError} on any non-success response.
 */
export const uploadCanvasImage = async (
  blob: Blob,
  options: UploadCanvasImageOptions = {}
): Promise<CanvasImageUploadResult> => {
  const fetchImpl = options.fetch ?? globalThis.fetch;

  const query = new URLSearchParams({
    image_category: options.imageCategory ?? 'other',
    is_intermediate: String(options.isIntermediate ?? false),
  });
  if (options.boardId) {
    query.set('board_id', options.boardId);
  }

  const fileName = options.fileName ?? 'canvas-paint.png';
  const file = new File([blob], fileName, { type: blob.type || 'image/png' });
  const body = new FormData();
  body.append('file', file);
  if (options.metadata) {
    body.append('metadata', JSON.stringify(options.metadata));
  }
  if (options.resizeTo) {
    body.append('resize_to', JSON.stringify(options.resizeTo));
  }

  const headers = new Headers();
  const token = getAuthToken();
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  let response: Response;
  try {
    response = await fetchImpl(buildApiUrl(`/api/v1/images/upload?${query.toString()}`), {
      body,
      headers,
      method: 'POST',
      signal: options.signal,
    });
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw error;
    }
    throw new CanvasImageUploadError(
      `Canvas image upload failed: ${error instanceof Error ? error.message : String(error)}`,
      null
    );
  }

  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new CanvasImageUploadError(
      text || `Canvas image upload failed: ${response.status} ${response.statusText}`,
      response.status
    );
  }

  const dto = (await response.json()) as UploadImageResponseDTO;
  return { height: dto.height, imageName: dto.image_name, width: dto.width };
};

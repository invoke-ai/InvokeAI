/**
 * Persist paint as imageName references. Category other hides pixels from gallery views; is_intermediate=false
 * makes referenced pixels durable. Only unreferenced generation composites are intermediate. Uploads use the
 * shared authentication and base-URL policy.
 */

import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';

import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { buildApiUrl, getHttpAuthToken } from '@platform/transport/http';

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
  /** Records which of the caller's projects the upload originates in, so intermediates can be cleared per project. */
  projectId?: string;
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

/** The server's refusal of a `project_id` it has no record of for this account (`assert_project_owned`). */
export const isUploadProjectNotFound = (error: unknown): boolean => {
  if (!(error instanceof CanvasImageUploadError) || error.status !== 404) {
    return false;
  }
  try {
    const body: unknown = JSON.parse(error.message);
    return typeof body === 'object' && body !== null && 'detail' in body && body.detail === 'Project not found';
  } catch {
    return false;
  }
};

export const uploadCanvasImage = async (
  blob: Blob,
  options: UploadCanvasImageOptions = {}
): Promise<CanvasImageUploadResult> => {
  const owner = captureAccountScope();
  const fetchImpl = options.fetch ?? globalThis.fetch;
  const signal = options.signal ? AbortSignal.any([options.signal, owner.signal]) : owner.signal;

  const query = new URLSearchParams({
    image_category: options.imageCategory ?? 'other',
    is_intermediate: String(options.isIntermediate ?? false),
  });
  if (options.boardId) {
    query.set('board_id', options.boardId);
  }
  if (options.projectId) {
    query.set('project_id', options.projectId);
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
  const token = getHttpAuthToken();
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  let response: Response;
  try {
    response = await fetchImpl(buildApiUrl(`/api/v1/images/upload?${query.toString()}`), {
      body,
      headers,
      method: 'POST',
      signal,
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

  assertAccountScopeCurrent(owner);
  if (!response.ok) {
    const text = await response.text().catch(() => '');

    assertAccountScopeCurrent(owner);
    throw new CanvasImageUploadError(
      text || `Canvas image upload failed: ${response.status} ${response.statusText}`,
      response.status
    );
  }

  const dto = (await response.json()) as UploadImageResponseDTO;

  assertAccountScopeCurrent(owner);
  return { height: dto.height, imageName: dto.image_name, width: dto.width };
};

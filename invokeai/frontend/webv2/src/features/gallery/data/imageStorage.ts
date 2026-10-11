import type { AccountScope } from '@platform/state/accountLifecycle';
import type { QueryClient } from '@tanstack/react-query';

import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { apiFetchJson } from '@platform/transport/http';
import { queryOptions } from '@tanstack/react-query';

import { invalidateGallery } from './queryCache';

/**
 * Server image storage: the subfolder layout new images are written to, and the admin-only move/recovery job
 * that relocates existing files into it. Both routes are admin-only, so callers gate reads on the capability.
 */

export const IMAGE_STORAGE_LAYOUTS = ['flat', 'date', 'type', 'hash'] as const;

export type ImageStorageLayout = (typeof IMAGE_STORAGE_LAYOUTS)[number];

export const isImageStorageLayout = (value: unknown): value is ImageStorageLayout =>
  IMAGE_STORAGE_LAYOUTS.includes(value as ImageStorageLayout);

export type ImageMoveOperation = 'move_all' | 'recovery';

export type ImageMoveJobState = 'planned' | 'moving' | 'moved' | 'committed' | 'error';

export interface ImageMoveJob {
  id: number;
  state: ImageMoveJobState;
  errorMessage: string | null;
}

export interface ImageMoveStatus {
  /** Whether a move or recovery is running right now. */
  isRunning: boolean;
  operation: ImageMoveOperation | null;
  /** A journaled job that has not committed; with `isRunning` false it needs recovery. */
  activeJobId: number | null;
  latestJob: ImageMoveJob | null;
  /** The last background worker error. */
  lastError: string | null;
  /** Images still stored under a layout other than the configured one. */
  needsMoveCount: number;
}

/** The job finished, one way or the other; its outcome is what `latestJob` reports. */
export const isTerminalImageMoveJobState = (state: ImageMoveJobState | null | undefined): boolean =>
  state === 'committed' || state === 'error';

/** `GET/PATCH /app/runtime_config` nest the settings under `config`; only the layout field is read here. */
interface RuntimeConfigDTO {
  config?: { image_subfolder_strategy?: unknown } | null;
}

interface ImageMoveJobDTO {
  id: number;
  state: ImageMoveJobState;
  error_message?: string | null;
}

interface ImageMoveStatusDTO {
  is_running: boolean;
  operation?: ImageMoveOperation | null;
  active_job_id?: number | null;
  latest_job?: ImageMoveJobDTO | null;
  last_error?: string | null;
  needs_move_count: number;
}

/** An unrecognised server value is kept, not coerced: the editor shows it as unknown instead of lying. */
export const readImageStorageLayout = (dto: RuntimeConfigDTO | null): string =>
  typeof dto?.config?.image_subfolder_strategy === 'string' ? dto.config.image_subfolder_strategy : 'flat';

const toImageMoveStatus = (dto: ImageMoveStatusDTO): ImageMoveStatus => ({
  activeJobId: dto.active_job_id ?? null,
  isRunning: dto.is_running,
  lastError: dto.last_error ?? null,
  latestJob: dto.latest_job
    ? { errorMessage: dto.latest_job.error_message ?? null, id: dto.latest_job.id, state: dto.latest_job.state }
    : null,
  needsMoveCount: dto.needs_move_count,
  operation: dto.operation ?? null,
});

const getAccountKey = (owner: AccountScope) => ({ accountId: owner.accountId, epoch: owner.epoch });

export const imageStorageKeys = {
  all: ['gallery', 'image-storage'] as const,
  layout: (owner: AccountScope) => [...imageStorageKeys.all, 'layout', getAccountKey(owner)] as const,
  moveStatus: (owner: AccountScope) => [...imageStorageKeys.all, 'move-status', getAccountKey(owner)] as const,
};

const fenced = <T>(owner: AccountScope, request: (signal: AbortSignal) => Promise<T>, signal: AbortSignal) =>
  request(AbortSignal.any([signal, owner.signal])).then((result) => {
    assertAccountScopeCurrent(owner);
    return result;
  });

export const getImageStorageLayout = (signal?: AbortSignal): Promise<string> =>
  apiFetchJson<RuntimeConfigDTO>('/api/v1/app/runtime_config', { signal }).then(readImageStorageLayout);

export const imageStorageLayoutQueryOptions = (owner = captureAccountScope()) =>
  queryOptions<string>({
    queryFn: ({ signal }) => fenced(owner, getImageStorageLayout, signal),
    queryKey: imageStorageKeys.layout(owner),
    staleTime: 30_000,
  });

/** Takes effect for new images at once; existing files move only through {@link startImageMove}. */
export const updateImageStorageLayout = async (
  client: QueryClient,
  layout: ImageStorageLayout,
  owner = captureAccountScope()
): Promise<void> => {
  const response = await apiFetchJson<RuntimeConfigDTO>('/api/v1/app/runtime_config', {
    body: JSON.stringify({ image_subfolder_strategy: layout }),
    method: 'PATCH',
    signal: owner.signal,
  });

  assertAccountScopeCurrent(owner);
  client.setQueryData(imageStorageKeys.layout(owner), readImageStorageLayout(response));
  // The count of images outside the layout changes with it.
  await client.invalidateQueries({ queryKey: imageStorageKeys.moveStatus(owner) });
};

export const getImageMoveStatus = (signal?: AbortSignal): Promise<ImageMoveStatus> =>
  apiFetchJson<ImageMoveStatusDTO>('/api/v1/image_moves/status', { signal }).then(toImageMoveStatus);

/**
 * Polls while a job runs. The read that first sees a job stop refreshes the gallery, since the move changed
 * where its files live, whichever consumer happens to hold the query at the time.
 */
export const imageMoveStatusQueryOptions = (owner = captureAccountScope()) =>
  queryOptions<ImageMoveStatus>({
    queryFn: async ({ client, signal }) => {
      const previous = client.getQueryData<ImageMoveStatus>(imageStorageKeys.moveStatus(owner));
      const status = await fenced(owner, getImageMoveStatus, signal);

      if (previous?.isRunning && !status.isRunning) {
        void invalidateGallery(client, owner);
      }

      return status;
    },
    queryKey: imageStorageKeys.moveStatus(owner),
    refetchInterval: (query) => (query.state.data?.isRunning ? 2_000 : false),
    // A move pauses the whole server; keep following it from a background tab so its end is not missed.
    refetchIntervalInBackground: true,
    staleTime: 5_000,
  });

const postImageMove = async (
  client: QueryClient,
  route: 'start' | 'recover',
  owner: AccountScope
): Promise<ImageMoveStatus> => {
  const status = await apiFetchJson<ImageMoveStatusDTO>(`/api/v1/image_moves/${route}`, {
    method: 'POST',
    signal: owner.signal,
  }).then(toImageMoveStatus);

  assertAccountScopeCurrent(owner);
  // The accepted status says the job is running, which starts the status query polling.
  client.setQueryData(imageStorageKeys.moveStatus(owner), status);

  return status;
};

/** Moves every image into the configured layout; the server refuses (409) while queue work or a job is active. */
export const startImageMove = (client: QueryClient, owner = captureAccountScope()): Promise<ImageMoveStatus> =>
  postImageMove(client, 'start', owner);

/** Resumes or finalizes a journaled job the server stopped in; refused (409) while a job runs. */
export const startImageMoveRecovery = (client: QueryClient, owner = captureAccountScope()): Promise<ImageMoveStatus> =>
  postImageMove(client, 'recover', owner);

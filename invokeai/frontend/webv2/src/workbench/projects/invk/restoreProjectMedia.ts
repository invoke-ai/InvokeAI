import type { AccountScope } from '@platform/state/accountLifecycle';

import { mapWithConcurrency } from '@platform/core/concurrency';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { ProjectCreateAbsentError } from '@workbench/projects/api';

import type { InvkBoardItem, InvkMediaKind } from './board';
import type { InvkMediaRef, ProjectTransferIssues } from './transfer';

import {
  deleteArchiveImages,
  deleteArchiveVideos,
  deleteStagingBoard,
  findExistingImageNames,
  findExistingVideoNames,
  INVK_TRANSFER_CONCURRENCY,
  isRequestCancellation,
  mimeForEntryName,
  starImages,
  starVideos,
  type UploadedImage,
  type UploadedVideo,
  uploadArchiveImage,
  uploadArchiveVideo,
} from './assetTransport';
import { buildMissingMediaName, createTransferIssueLog, toMediaKey } from './transfer';

/** Failed names always remap away from existing media; loss reporting counts only live references. */

/** Ancillary/probe failures are best-effort; operation cancellation propagates. */
const degradeUnlessCancelled =
  <T>(fallback: T) =>
  (error: unknown): T => {
    if (isRequestCancellation(error)) {
      throw error;
    }

    return fallback;
  };

export interface MaterializedMedia {
  kind: InvkMediaKind;
  name: string;
  /** The name the exporting server used, which the document still refers to. */
  sourceName: string;
}

export interface MaterializeFailure {
  kind: InvkMediaKind;
  name: string;
  reason: 'missing-entry' | 'upload-failed';
}

export interface MaterializeResult {
  failed: MaterializeFailure[];
  materialized: MaterializedMedia[];
}

/** Takes every board at once so small boards are scheduled together; onItemSettled fires once per descriptor. */
export type MediaMaterializer = (
  boards: readonly RestoreBoardInput[],
  onItemSettled: () => void
) => Promise<MaterializeResult>;

/** Filled in as the restore runs, so a restore that throws part way can still be undone. */
export interface RestoredMediaLedger {
  boardIds: string[];
  boardImageNames: string[];
  boardVideoNames: string[];
  coverImageName: string | null;
  imageNames: string[];
  videoNames: string[];
}

export const createRestoredMediaLedger = (boardIds: readonly string[]): RestoredMediaLedger => ({
  boardIds: [...boardIds],
  boardImageNames: [],
  boardVideoNames: [],
  coverImageName: null,
  imageNames: [],
  videoNames: [],
});

export interface RestoreBoardInput {
  items: readonly InvkBoardItem[];
  stagingBoardId: string;
}

export interface RestoreProjectMediaInput {
  /** The source's boards, inbox first; an item is on exactly one of them. */
  boards: readonly RestoreBoardInput[];
  coverBytes: { bytes: Uint8Array; entryName: string } | null;
  coverSourceImageName: string | null;
  documentRefs: readonly InvkMediaRef[];
  ledger: RestoredMediaLedger;
  projectId: string;
}

export interface RestoreProjectMediaDeps {
  documentMediaBytes?: (ref: InvkMediaRef) => Uint8Array | undefined;
  findExistingImageNames?: (imageNames: readonly string[], signal?: AbortSignal) => Promise<Set<string>>;
  findExistingVideoNames?: (videoNames: readonly string[], signal?: AbortSignal) => Promise<Set<string>>;
  materializeBoardMedia: MediaMaterializer;
  onProgress?: (progress: { completed: number; total: number }) => void;
  signal?: AbortSignal;
  starImages?: (imageNames: readonly string[], signal?: AbortSignal) => Promise<{ failed: string[] }>;
  starVideos?: (videoNames: readonly string[], signal?: AbortSignal) => Promise<{ failed: string[] }>;
  uploadImage?: (
    bytes: Uint8Array,
    fileName: string,
    options?: { contentType?: string; signal?: AbortSignal }
  ) => Promise<UploadedImage>;
  uploadVideo?: (
    bytes: Uint8Array,
    fileName: string,
    options?: { contentType?: string; signal?: AbortSignal }
  ) => Promise<UploadedVideo>;
}

export interface RestoreProjectMediaResult extends ProjectTransferIssues {
  coverImageName: string | null;
  mappings: { images: Map<string, string>; videos: Map<string, string> };
}

interface PendingUpload {
  /** Cleared once uploaded, so a queue of hundreds does not pin every asset's bytes to the end. */
  bytes: Uint8Array | null;
  kind: InvkMediaKind;
  name: string;
}

interface RestoreKindAdapter {
  addBoardIdentity: (name: string) => void;
  addDocumentIdentity: (name: string) => void;
  mapping: Map<string, string>;
  markResolvable: (name: string) => void;
  upload: (bytes: Uint8Array, name: string, signal?: AbortSignal) => Promise<string>;
}

const resolveCoverImageName = async (input: {
  coverBytes: { bytes: Uint8Array; entryName: string } | null;
  coverSourceImageName: string | null;
  ledger: RestoredMediaLedger;
  mappings: { images: ReadonlyMap<string, string> };
  restoredImageNames: ReadonlySet<string>;
  signal?: AbortSignal;
  uploadImage: NonNullable<RestoreProjectMediaDeps['uploadImage']>;
}): Promise<string | null> => {
  if (input.coverSourceImageName !== null) {
    const restoredName = input.mappings.images.get(input.coverSourceImageName) ?? input.coverSourceImageName;

    if (input.restoredImageNames.has(restoredName)) {
      return restoredName;
    }
  }

  if (input.coverBytes === null) {
    return null;
  }

  try {
    const uploaded = await input.uploadImage(input.coverBytes.bytes, input.coverBytes.entryName, {
      contentType: mimeForEntryName(input.coverBytes.entryName),
      signal: input.signal,
    });

    input.ledger.coverImageName = uploaded.imageName;

    return uploaded.imageName;
  } catch {
    return null;
  }
};

/** Per-media failures belong in the result; rejections end the whole operation. */
export const restoreProjectMedia = async (
  input: RestoreProjectMediaInput,
  deps: RestoreProjectMediaDeps
): Promise<RestoreProjectMediaResult> => {
  const checkExistingImages = deps.findExistingImageNames ?? findExistingImageNames;
  const checkExistingVideos = deps.findExistingVideoNames ?? findExistingVideoNames;
  const uploadImage = deps.uploadImage ?? uploadArchiveImage;
  const uploadVideo = deps.uploadVideo ?? uploadArchiveVideo;
  const starRestoredImages = deps.starImages ?? starImages;
  const starRestoredVideos = deps.starVideos ?? starVideos;

  const issues = createTransferIssueLog();
  const mappings = { images: new Map<string, string>(), videos: new Map<string, string>() };
  const documentKeys = new Set(input.documentRefs.map(toMediaKey));
  const boardItems = input.boards.flatMap((board) => board.items);
  const boardKeys = new Set(boardItems.map(toMediaKey));
  /** Descriptor position, so a placeholder name does not depend on the order failures happened in. */
  const descriptorIndexes = new Map(boardItems.map((item, index) => [toMediaKey(item), index]));
  let nextUnknownIndex = boardItems.length;
  const missingNameIndex = (key: string): number => {
    const index = descriptorIndexes.get(key);

    if (index !== undefined) {
      return index;
    }

    nextUnknownIndex += 1;

    return nextUnknownIndex - 1;
  };
  const starredKeys = new Set(boardItems.filter((item) => item.starred).map(toMediaKey));

  const documentOnlyRefs = input.documentRefs.filter((ref) => !boardKeys.has(toMediaKey(ref)));
  const documentOnlyImages = documentOnlyRefs.filter((ref) => ref.kind === 'image').map((ref) => ref.name);
  const documentOnlyVideos = documentOnlyRefs.filter((ref) => ref.kind === 'video').map((ref) => ref.name);

  const stagedBoards = input.boards.filter((board) => board.items.length > 0);

  let completed = 0;
  let total =
    stagedBoards.reduce((count, board) => count + board.items.length, 0) +
    documentOnlyRefs.length +
    (input.coverBytes === null ? 0 : 1);
  const advance = (): void => {
    completed += 1;
    deps.onProgress?.({ completed, total });
  };

  const restoredImageNames = new Set<string>();
  const kindAdapters: Record<InvkMediaKind, RestoreKindAdapter> = {
    image: {
      addBoardIdentity: (name) => input.ledger.boardImageNames.push(name),
      addDocumentIdentity: (name) => input.ledger.imageNames.push(name),
      mapping: mappings.images,
      markResolvable: (name) => restoredImageNames.add(name),
      upload: async (bytes, name, signal) =>
        (await uploadImage(bytes, name, { contentType: mimeForEntryName(name), signal })).imageName,
    },
    video: {
      addBoardIdentity: (name) => input.ledger.boardVideoNames.push(name),
      addDocumentIdentity: (name) => input.ledger.videoNames.push(name),
      mapping: mappings.videos,
      markResolvable: () => undefined,
      upload: async (bytes, name, signal) =>
        (await uploadVideo(bytes, name, { contentType: mimeForEntryName(name, 'video'), signal })).videoName,
    },
  };

  // Attach allSettled now to prevent unhandled rejection; cancellation propagates at the ordered await.
  const settledProbes = Promise.allSettled([
    documentOnlyImages.length === 0
      ? Promise.resolve(new Set<string>())
      : checkExistingImages(documentOnlyImages, deps.signal).catch(degradeUnlessCancelled(new Set<string>())),
    documentOnlyVideos.length === 0
      ? Promise.resolve(new Set<string>())
      : checkExistingVideos(documentOnlyVideos, deps.signal).catch(degradeUnlessCancelled(new Set<string>())),
  ]);

  const boardResult = await deps.materializeBoardMedia(stagedBoards, advance);

  const starTargets = { image: [] as string[], video: [] as string[] };
  const sourceNamesByFreshName = new Map<string, string>();
  const settledBoardKeys = new Set<string>();

  for (const entry of boardResult.materialized) {
    const key = toMediaKey({ kind: entry.kind, name: entry.sourceName });
    const adapter = kindAdapters[entry.kind];

    settledBoardKeys.add(key);
    sourceNamesByFreshName.set(toMediaKey({ kind: entry.kind, name: entry.name }), entry.sourceName);

    adapter.addBoardIdentity(entry.name);
    adapter.markResolvable(entry.name);

    if (entry.sourceName !== entry.name) {
      adapter.mapping.set(entry.sourceName, entry.name);
    }

    if (starredKeys.has(key)) {
      starTargets[entry.kind].push(entry.name);
    }
  }

  const failBoardItem = (failure: MaterializeFailure): void => {
    const key = toMediaKey(failure);

    if (settledBoardKeys.has(key)) {
      return;
    }

    settledBoardKeys.add(key);
    issues.addBoardItemIssue(failure, failure.reason);

    kindAdapters[failure.kind].mapping.set(
      failure.name,
      buildMissingMediaName(input.projectId, failure.kind, missingNameIndex(key))
    );

    if (documentKeys.has(key)) {
      issues.addDocumentReferenceIssue(failure, failure.reason);
    }
  };

  for (const failure of boardResult.failed) {
    failBoardItem(failure);
  }

  for (const item of boardItems) {
    if (!settledBoardKeys.has(toMediaKey(item))) {
      failBoardItem({ kind: item.kind, name: item.name, reason: 'upload-failed' });
    }
  }

  // Star failure must not trigger rollback of successfully restored media.
  const [imageStars, videoStars] = await Promise.all([
    starRestoredImages(starTargets.image, deps.signal).catch(degradeUnlessCancelled({ failed: starTargets.image })),
    starRestoredVideos(starTargets.video, deps.signal).catch(degradeUnlessCancelled({ failed: starTargets.video })),
  ]);

  for (const [kind, failed] of [
    ['image', imageStars.failed],
    ['video', videoStars.failed],
  ] as const) {
    for (const freshName of failed) {
      const sourceName = sourceNamesByFreshName.get(toMediaKey({ kind, name: freshName })) ?? freshName;

      issues.addBoardItemIssue({ kind, name: sourceName }, 'star-failed');
    }
  }

  const [existingImages, existingVideos] = (await settledProbes).map((settled) => {
    if (settled.status === 'rejected') {
      throw settled.reason;
    }

    return settled.value;
  }) as [Set<string>, Set<string>];

  const pending: PendingUpload[] = [];
  const plannedTotal = total;

  for (const [names, existing, kind] of [
    [documentOnlyImages, existingImages, 'image'],
    [documentOnlyVideos, existingVideos, 'video'],
  ] as const) {
    for (const name of names) {
      if (existing.has(name)) {
        kindAdapters[kind].markResolvable(name);

        total -= 1;
        continue;
      }

      const bytes = deps.documentMediaBytes?.({ kind, name });

      if (bytes === undefined) {
        issues.addDocumentReferenceIssue({ kind, name }, 'missing-entry');
        total -= 1;
        continue;
      }

      pending.push({ bytes, kind, name });
    }
  }

  if (total !== plannedTotal) {
    deps.onProgress?.({ completed, total });
  }

  await mapWithConcurrency(pending, INVK_TRANSFER_CONCURRENCY, async (item) => {
    const { kind, name } = item;
    const bytes = item.bytes;

    // Release queued bytes before upload; the request body retains them.
    item.bytes = null;

    if (bytes === null) {
      return;
    }

    try {
      const adapter = kindAdapters[kind];
      const restoredName = await adapter.upload(bytes, name, deps.signal);

      adapter.addDocumentIdentity(restoredName);
      adapter.markResolvable(restoredName);

      if (restoredName !== name) {
        adapter.mapping.set(name, restoredName);
      }
    } catch (error) {
      // Cancellation is one operation failure, not a missing-asset result for every queued item.
      if (isRequestCancellation(error)) {
        throw error;
      }

      // Failed uploads must remain unresolved rather than bind to unrelated existing media.
      issues.addDocumentReferenceIssue({ kind, name }, 'upload-failed');
    }

    advance();
  });

  const coverImageName = await resolveCoverImageName({
    coverBytes: input.coverBytes,
    coverSourceImageName: input.coverSourceImageName,
    ledger: input.ledger,
    mappings,
    restoredImageNames,
    ...(deps.signal === undefined ? {} : { signal: deps.signal }),
    uploadImage,
  });

  if (input.coverBytes !== null) {
    advance();
  }

  return { coverImageName, mappings, ...issues.toIssues() };
};

/** Rollback only when creation failed, `ProjectCreateAbsentError` proves absence, and the account is unchanged. */
export const rollbackUnlessProjectExists = async (
  error: unknown,
  didCreateProject: boolean,
  owner: AccountScope,
  rollback: () => Promise<void>
): Promise<void> => {
  if (didCreateProject || !(error instanceof ProjectCreateAbsentError) || !isAccountScopeCurrent(owner)) {
    return;
  }

  try {
    await rollback();
  } catch {
    // Best-effort: unknown outcomes never authorize deletion.
  }
};

export interface RollbackRestoredMediaDeps {
  deleteBoard?: (boardId: string, signal?: AbortSignal) => Promise<void>;
  deleteImages?: (imageNames: string[], signal?: AbortSignal) => Promise<void>;
  deleteVideos?: (videoNames: string[], signal?: AbortSignal) => Promise<void>;
  signal?: AbortSignal;
}

/** Best-effort cleanup deletes only ledger-owned resources, then the board without include_images. */
export const rollbackRestoredMedia = async (
  ledger: RestoredMediaLedger,
  deps: RollbackRestoredMediaDeps = {}
): Promise<void> => {
  const deleteImages = deps.deleteImages ?? deleteArchiveImages;
  const deleteVideos = deps.deleteVideos ?? deleteArchiveVideos;
  const deleteBoard = deps.deleteBoard ?? deleteStagingBoard;
  const imageNames = [
    ...ledger.boardImageNames,
    ...ledger.imageNames,
    ...(ledger.coverImageName === null ? [] : [ledger.coverImageName]),
  ].sort();
  const videoNames = [...ledger.boardVideoNames, ...ledger.videoNames].sort();

  await Promise.allSettled([
    imageNames.length === 0 ? Promise.resolve() : deleteImages(imageNames, deps.signal),
    videoNames.length === 0 ? Promise.resolve() : deleteVideos(videoNames, deps.signal),
  ]);

  await Promise.allSettled(
    // A staging board left behind is a deletable Library board, not a broken state.
    ledger.boardIds.map((boardId) => deleteBoard(boardId, deps.signal))
  );
};

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

/**
 * Putting a project's media on this server, for whichever direction needs it. Import and
 * duplication differ in exactly one step — where the bytes come from, {@link MediaMaterializer} —
 * and share everything around it.
 *
 * ### A failed board item must not resolve to a stranger
 *
 * An item both on the board and named by the document, whose materialization fails, must not keep
 * its old name: on the same server that name is already taken, by the *source* project's image, so
 * the project would open showing the right picture from the wrong owner. Failed items are forced
 * onto {@link buildMissingMediaName}, which resolves to nothing and renders as a missing layer.
 *
 * The placeholder is written for every failed item, because `remapAssetRefs` walks the whole
 * document. The live reference set — which skips history — decides only what is worth *reporting*:
 * a gallery recent that stops resolving is not something the person lost from this project.
 */

/**
 * Degrade to `fallback`, except for cancellation. Starring is a flag and an existence probe only
 * spares an upload — neither is worth reaching the caller's rollback. Cancellation is the
 * exception because it makes *every* remaining call fail, which is not a half-restored project.
 */
const degradeUnlessCancelled =
  <T>(fallback: T) =>
  (error: unknown): T => {
    if (isRequestCancellation(error)) {
      throw error;
    }

    return fallback;
  };

/** One board item that now exists here, under the name this server chose for it. */
export interface MaterializedMedia {
  kind: InvkMediaKind;
  /** The authoritative identity this server assigned. */
  name: string;
  /** The name the exporting server used, which the document still refers to. */
  sourceName: string;
}

export interface MaterializeFailure {
  kind: InvkMediaKind;
  /** The source name, so a caller can match it against the descriptor it came from. */
  name: string;
  reason: 'missing-entry' | 'upload-failed';
}

export interface MaterializeResult {
  failed: MaterializeFailure[];
  materialized: MaterializedMedia[];
}

/**
 * Make board media exist on `boardId`, however this direction gets its bytes. `onItemSettled` fires
 * once per descriptor, so progress moves through the part people actually wait for.
 */
export type MediaMaterializer = (
  items: readonly InvkBoardItem[],
  boardId: string,
  onItemSettled: () => void
) => Promise<MaterializeResult>;

/**
 * Every identity a restore created, so a failure can undo exactly what it did. Filled in as it
 * runs, not returned at the end: a restore that throws part way has still created things.
 */
export interface RestoredMediaLedger {
  /** The staging board these were created on, when this restore created one. */
  boardId: string | null;
  boardImageNames: string[];
  boardVideoNames: string[];
  /** A cover thumbnail uploaded as a fallback, unboarded. */
  coverImageName: string | null;
  /** Unboarded uploads made to satisfy document-only references. */
  imageNames: string[];
  videoNames: string[];
}

export const createRestoredMediaLedger = (boardId: string | null): RestoredMediaLedger => ({
  boardId,
  boardImageNames: [],
  boardVideoNames: [],
  coverImageName: null,
  imageNames: [],
  videoNames: [],
});

export interface RestoreProjectMediaInput {
  /** The staging board to materialize onto, or `null` when this project has no board media. */
  boardId: string | null;
  /** The board's contents as the source enumerated them, canonically ordered. */
  boardItems: readonly InvkBoardItem[];
  /** Bundled cover bytes, for the case where the cover's source image cannot be restored. */
  coverBytes: { bytes: Uint8Array; entryName: string } | null;
  /** The image the document nominates as its cover, under its pre-restore name. */
  coverSourceImageName: string | null;
  /** Live references collected from the canonical document. */
  documentRefs: readonly InvkMediaRef[];
  /** Written as the restore runs; the caller keeps it for rollback. */
  ledger: RestoredMediaLedger;
  /** The id the project will be created under. Placeholder names derive from it. */
  projectId: string;
}

export interface RestoreProjectMediaDeps {
  /** Bytes for a document-only reference this server is missing, when the source carries any. */
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
  /** The image to record as this project's cover, or `null` for a project that has none. */
  coverImageName: string | null;
  /** Old name to new name, per kind, for the document rewrite. */
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

/**
 * The cover, preferring an image the restore already put here — the same picture without a second
 * copy. Uploading the bundled entry unconditionally left one orphan per import, in the private
 * `'other'` category where nobody could find or delete it. The bytes are the fallback, for a cover
 * whose source image is dangling.
 */
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
    // A project with no cover shows the folder glyph, which is a state the library already
    // renders. Not worth failing an import over.
    return null;
  }
};

/**
 * Restore a project's media and report what could not be carried. Nothing throws for a media
 * failure; the only rejections are the ones that end the operation — cancellation, an expired
 * account.
 */
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
  const boardKeys = new Set(input.boardItems.map(toMediaKey));
  /** Descriptor position, so a placeholder name does not depend on the order failures happened in. */
  const descriptorIndexes = new Map(input.boardItems.map((item, index) => [toMediaKey(item), index]));
  /**
   * Descriptor position where it exists; counting past the descriptors otherwise. Collapsing
   * position-less failures onto one index would merge two unrelated missing items into a single
   * dangling reference.
   */
  let nextUnknownIndex = input.boardItems.length;
  const missingNameIndex = (key: string): number => {
    const index = descriptorIndexes.get(key);

    if (index !== undefined) {
      return index;
    }

    nextUnknownIndex += 1;

    return nextUnknownIndex - 1;
  };
  const starredKeys = new Set(input.boardItems.filter((item) => item.starred).map(toMediaKey));

  // Document-only references are the ones the board does not own; those the board owns are
  // restored as board media and their references follow the copy.
  const documentOnlyRefs = input.documentRefs.filter((ref) => !boardKeys.has(toMediaKey(ref)));
  const documentOnlyImages = documentOnlyRefs.filter((ref) => ref.kind === 'image').map((ref) => ref.name);
  const documentOnlyVideos = documentOnlyRefs.filter((ref) => ref.kind === 'video').map((ref) => ref.name);

  // A board with no staging board to put it on cannot be materialized; every descriptor then falls
  // through to the unsettled pass below, which is the honest reading of "the media did not arrive".
  const stagingBoardId = input.boardItems.length === 0 ? null : input.boardId;

  let completed = 0;
  // The document-only total is only known once the existence checks answer, so progress is reported
  // against the worst case: every reference needing an upload. It can only finish early.
  let total =
    (stagingBoardId === null ? 0 : input.boardItems.length) +
    documentOnlyRefs.length +
    (input.coverBytes === null ? 0 : 1);
  const advance = (): void => {
    completed += 1;
    deps.onProgress?.({ completed, total });
  };

  /** Names this server can serve once the restore is done, under their final identities. */
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

  // Started before the board upload, awaited after it: the probes need only the document's own
  // references, and the board upload is the minutes-long part. An empty answer degrades safely —
  // the check only skips redundant uploads, so failing it costs bandwidth, never correctness.
  //
  // `allSettled` is constructed here, not at the await: a promise started now and awaited after the
  // upload would surface as an unhandled rejection if the upload threw first. Cancellation still
  // ends the restore, re-thrown at the await below in its proper order.
  const settledProbes = Promise.allSettled([
    documentOnlyImages.length === 0
      ? Promise.resolve(new Set<string>())
      : checkExistingImages(documentOnlyImages, deps.signal).catch(degradeUnlessCancelled(new Set<string>())),
    documentOnlyVideos.length === 0
      ? Promise.resolve(new Set<string>())
      : checkExistingVideos(documentOnlyVideos, deps.signal).catch(degradeUnlessCancelled(new Set<string>())),
  ]);

  const boardResult =
    stagingBoardId === null
      ? ({ failed: [], materialized: [] } satisfies MaterializeResult)
      : await deps.materializeBoardMedia(input.boardItems, stagingBoardId, advance);

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

  /** A board item that did not arrive: forced dangling everywhere, and reported where it shows. */
  const failBoardItem = (failure: MaterializeFailure): void => {
    const key = toMediaKey(failure);

    if (settledBoardKeys.has(key)) {
      return;
    }

    settledBoardKeys.add(key);
    issues.addBoardItemIssue(failure, failure.reason);

    // Mapped unconditionally: `remapAssetRefs` rewrites the whole document while `documentKeys`
    // covers only live references, so a name left unmapped survives in history — pointing, on this
    // server, at the source project's own image.
    kindAdapters[failure.kind].mapping.set(
      failure.name,
      buildMissingMediaName(input.projectId, failure.kind, missingNameIndex(key))
    );

    // Only the live references are *reported*. A gallery recent that no longer resolves is not
    // something the person lost from this project, and counting it would inflate every report.
    if (documentKeys.has(key)) {
      issues.addDocumentReferenceIssue(failure, failure.reason);
    }
  };

  for (const failure of boardResult.failed) {
    failBoardItem(failure);
  }

  // A descriptor reported neither way is a failure too: dropping it silently would leave its
  // document reference on the old name.
  for (const item of input.boardItems) {
    if (!settledBoardKeys.has(toMediaKey(item))) {
      failBoardItem({ kind: item.kind, name: item.name, reason: 'upload-failed' });
    }
  }

  // A failed star costs a flag, not the media, so it must not reject: a rejection here reaches the
  // caller's rollback and deletes every image just uploaded, over a flag.
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

        // Satisfied by media that is already here, which is the whole point of the dedup.
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

  // Deduplication and missing bytes both shrink the work; say so, or a restore that ends with
  // nothing left to upload reports its last count against a total it will never reach.
  if (total !== plannedTotal) {
    deps.onProgress?.({ completed, total });
  }

  await mapWithConcurrency(pending, INVK_TRANSFER_CONCURRENCY, async (item) => {
    const { kind, name } = item;
    const bytes = item.bytes;

    // Dropped before the request, not after: the upload body holds its own reference, and this
    // queue was why a large restore held every asset it had not sent yet.
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
      // Not a failure of this asset: cancellation makes every remaining upload fail too, and
      // reporting them one by one would name hundreds of dangling references.
      if (isRequestCancellation(error)) {
        throw error;
      }

      // A failed upload leaves the reference pointing at a name this server does not have — the
      // same outcome as an asset the source never carried, and honest for the same reason.
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

/**
 * Run a restore's rollback, but only when the project it was staging certainly does not exist.
 * Three things must hold: the create did not already succeed; the failure *proves* absence, which
 * is what {@link ProjectCreateAbsentError} means and why the create goes through
 * `createProjectSettled`; and the account has not changed. An unknown outcome must never authorize
 * a deletion — uploads are recoverable clutter, a project stripped of its media is not.
 */
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
    // Best-effort, per the docblock above.
  }
};

export interface RollbackRestoredMediaDeps {
  deleteBoard?: (boardId: string, signal?: AbortSignal) => Promise<void>;
  deleteImages?: (imageNames: string[], signal?: AbortSignal) => Promise<void>;
  deleteVideos?: (videoNames: string[], signal?: AbortSignal) => Promise<void>;
  signal?: AbortSignal;
}

/**
 * Undo a restore whose project was never created — exactly the identities it made, then the board.
 * Best-effort throughout, so a failing cleanup cannot replace the actionable error. The board goes
 * last and without `include_images`; see {@link deleteStagingBoard}.
 */
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

  if (ledger.boardId === null) {
    return;
  }

  try {
    await deleteBoard(ledger.boardId, deps.signal);
  } catch {
    // An unclaimed private board is invisible clutter, not a broken state.
  }
};

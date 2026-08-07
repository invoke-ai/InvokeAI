import { mapWithConcurrency } from '@platform/core/concurrency';

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
 * Putting a project's media on this server, for whichever direction needs it.
 *
 * Import and duplication ask the same question — "make this project's media exist here, under
 * identities this project owns" — and differ in exactly one step: where the bytes come from.
 * Import has them in the archive; duplication has them already on this server and copies them
 * without a byte ever reaching the browser. That step is {@link MediaMaterializer}, and everything
 * around it — planning, remapping, starring, issue reporting, the rollback ledger — is shared.
 *
 * ### Board media always gets a new identity; document references may be reused
 *
 * `board_images` has `PRIMARY KEY (image_name)`: an image sits on exactly one board. A restore that
 * saw its own name already on the destination and reused it would not be restoring the project's
 * board, it would be *moving* somebody else's image onto it — and deleting either project's board
 * would then take that image from both. So every board item is materialized fresh, with no
 * existence check, and the document is rewritten to the new name.
 *
 * Document-only references are pointers rather than membership, so an image the destination already
 * has satisfies them. That is the v2 behaviour and it is still right: it is what makes re-importing
 * your own export nearly free instead of doubling your disk.
 *
 * ### A failed board item must not resolve to a stranger
 *
 * The dangerous case is an item that is *both* on the board and referenced by the document, whose
 * materialization fails. Leaving the reference on the old name is not "no worse than before": on the
 * same server — every duplication, and every re-import — that name is already taken, by the source
 * project's own image. The project would open showing the right picture from the wrong owner, and
 * deleting the original would break it with no explanation. So a failed board item forces its
 * document references onto {@link buildMissingMediaName}, which resolves to nothing and renders as
 * a missing layer, exactly as a dangling v2 reference already does.
 */

/**
 * A `catch` handler that degrades to `fallback`, except for the errors that must end the restore.
 *
 * Nothing in a restore is worth the whole restore. Starring is a flag; an existence probe only
 * spares a redundant upload. A batch of either failing should cost what it is worth, not reach the
 * caller's rollback and delete every asset that had already arrived.
 *
 * Cancellation is the exception, and has to be, because it is not a failure of the call: an aborted
 * signal or a rotated account makes *every* remaining call fail. Degrading those would report the
 * project as half-restored and then create it anyway.
 */
/**
 * Hold a check's outcome — value or failure — so an early-started promise cannot become an
 * unhandled rejection while the restore is busy elsewhere.
 */
type SettledExisting = { names: Set<string> } | { error: unknown };

const settleExisting = (promise: Promise<Set<string>>): Promise<SettledExisting> =>
  promise.then(
    (names) => ({ names }),
    (error: unknown) => ({ error })
  );

const unwrapExisting = (settled: SettledExisting): Set<string> => {
  if ('error' in settled) {
    throw settled.error;
  }

  return settled.names;
};

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
 * Make board media exist on `boardId`, however this direction gets its bytes.
 *
 * `onItemSettled` is called once per descriptor, succeeded or failed. Restoring a board is the bulk
 * of an import's wall time, so a progress count that only moved once the whole board was done would
 * sit at zero through the part people actually wait for.
 */
export type MediaMaterializer = (
  items: readonly InvkBoardItem[],
  boardId: string,
  onItemSettled: () => void
) => Promise<MaterializeResult>;

/**
 * Every identity a restore created, so a failure can undo exactly what it did and nothing else.
 *
 * Filled in as the restore runs rather than returned at the end: a restore that throws part way —
 * a cancelled signal, an account rotation — has still created things, and a ledger only handed back
 * on success would leave them behind.
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

const emptyMaterializeResult = (): MaterializeResult => ({ failed: [], materialized: [] });

/**
 * The cover, preferring an image the restore already put on this server.
 *
 * The archive's `cover` entry is a thumbnail of an image the document also references, and
 * `getProjectCoverUrl` asks for a thumbnail anyway — so once that image is restored, pointing the
 * index at it is the same picture without a second copy. Uploading the entry unconditionally
 * instead left one orphan per import: covers go up under the canvas's private `'other'` category,
 * which appears in no gallery view and no board count, so nobody could ever find or delete them.
 *
 * The bundled bytes are the fallback, for the case that made them worth carrying: a cover whose
 * source image is dangling here.
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
 * Restore a project's media and report what could not be carried.
 *
 * Nothing here throws for a media failure — a project that opens with two broken layers is worth
 * far more than no project at all — so the only rejections are the ones that end the whole
 * operation: a cancelled signal, an expired account.
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

  // Started before the board upload, awaited after it. The existence checks depend only on the
  // document's own references, which are known here — and the board upload is the minutes-long part
  // of a large import, so waiting it out first meant the video probe, which is one request per
  // name, began only once everything else had finished.
  //
  // An empty answer is the safe degradation: the check exists only to skip uploading something the
  // destination already has, so failing it costs bandwidth, never correctness. Aborting the import
  // over it would be losing the project to save a request.
  //
  // Settled rather than left to reject: a promise started here but not awaited until after the
  // board upload would surface as an unhandled rejection if the upload threw first. Cancellation
  // still ends the restore — it is re-thrown at the await below, in its proper order.
  const existingImagesPromise = settleExisting(
    documentOnlyImages.length === 0
      ? Promise.resolve(new Set<string>())
      : checkExistingImages(documentOnlyImages, deps.signal).catch(degradeUnlessCancelled(new Set<string>()))
  );
  const existingVideosPromise = settleExisting(
    documentOnlyVideos.length === 0
      ? Promise.resolve(new Set<string>())
      : checkExistingVideos(documentOnlyVideos, deps.signal).catch(degradeUnlessCancelled(new Set<string>()))
  );

  const boardResult =
    stagingBoardId === null
      ? emptyMaterializeResult()
      : await deps.materializeBoardMedia(input.boardItems, stagingBoardId, advance);

  const starTargets = { image: [] as string[], video: [] as string[] };
  const sourceNamesByFreshName = new Map<string, string>();
  const settledBoardKeys = new Set<string>();

  for (const entry of boardResult.materialized) {
    const key = toMediaKey({ kind: entry.kind, name: entry.sourceName });

    settledBoardKeys.add(key);
    sourceNamesByFreshName.set(toMediaKey({ kind: entry.kind, name: entry.name }), entry.sourceName);

    if (entry.kind === 'image') {
      input.ledger.boardImageNames.push(entry.name);
      restoredImageNames.add(entry.name);
    } else {
      input.ledger.boardVideoNames.push(entry.name);
    }

    if (entry.sourceName !== entry.name) {
      mappings[entry.kind === 'image' ? 'images' : 'videos'].set(entry.sourceName, entry.name);
    }

    if (starredKeys.has(key)) {
      starTargets[entry.kind].push(entry.name);
    }
  }

  /** A board item that did not arrive: reported, and forced dangling if the document wanted it. */
  const failBoardItem = (failure: MaterializeFailure): void => {
    settledBoardKeys.add(toMediaKey(failure));
    issues.addBoardItemIssue(failure, failure.reason);

    const key = toMediaKey(failure);

    if (!documentKeys.has(key)) {
      return;
    }

    mappings[failure.kind === 'image' ? 'images' : 'videos'].set(
      failure.name,
      buildMissingMediaName(input.projectId, failure.kind, descriptorIndexes.get(key) ?? 0)
    );
    issues.addDocumentReferenceIssue(failure, failure.reason);
  };

  for (const failure of boardResult.failed) {
    failBoardItem(failure);
  }

  // A descriptor the materializer reported neither way is a failure too. Silently dropping it would
  // leave its document reference on the old name, which is the one outcome this module exists to
  // prevent.
  for (const item of input.boardItems) {
    if (!settledBoardKeys.has(toMediaKey(item))) {
      failBoardItem({ kind: item.kind, name: item.name, reason: 'upload-failed' });
    }
  }

  // Starring is a separate call by design: the same one import and duplication use, and the same
  // one the gallery uses. A star that fails costs a flag, not the media, so it is reported against
  // the board item and nowhere else — the document does not care whether a layer is starred.
  //
  // Which is why the call itself must not reject: a rejection here reaches the caller's rollback
  // and deletes every image that was just uploaded, over a flag. A whole batch that failed to send
  // is every name in it failing to be starred.
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

  const [existingImages, existingVideos] = (await Promise.all([existingImagesPromise, existingVideosPromise])).map(
    unwrapExisting
  ) as [Set<string>, Set<string>];

  const pending: PendingUpload[] = [];
  const plannedTotal = total;

  for (const [names, existing, kind] of [
    [documentOnlyImages, existingImages, 'image'],
    [documentOnlyVideos, existingVideos, 'video'],
  ] as const) {
    for (const name of names) {
      if (existing.has(name)) {
        if (kind === 'image') {
          restoredImageNames.add(name);
        }

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

    // Dropped from the queue before the request, not after: the upload body holds its own
    // reference for as long as it needs one, and the queue was the reason a restore of a large
    // project held every asset it had not sent yet.
    item.bytes = null;

    if (bytes === null) {
      return;
    }

    try {
      const options = { contentType: mimeForEntryName(name, kind), signal: deps.signal };
      const restoredName =
        kind === 'image'
          ? (await uploadImage(bytes, name, options)).imageName
          : (await uploadVideo(bytes, name, options)).videoName;

      if (kind === 'image') {
        input.ledger.imageNames.push(restoredName);
        restoredImageNames.add(restoredName);
      } else {
        input.ledger.videoNames.push(restoredName);
      }

      if (restoredName !== name) {
        mappings[kind === 'image' ? 'images' : 'videos'].set(name, restoredName);
      }
    } catch (error) {
      // Not a failure of this asset: an aborted signal or an expired account makes every remaining
      // upload fail too, and reporting them one by one would finish with a warning naming hundreds
      // of dangling references and a project created anyway.
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

export interface RollbackRestoredMediaDeps {
  deleteBoard?: (boardId: string, signal?: AbortSignal) => Promise<void>;
  deleteImages?: (imageNames: string[], signal?: AbortSignal) => Promise<void>;
  deleteVideos?: (videoNames: string[], signal?: AbortSignal) => Promise<void>;
  signal?: AbortSignal;
}

/**
 * Undo a restore whose project was never created — exactly the identities it made, then the board
 * it made them on.
 *
 * Best-effort throughout: the failure that brought us here is the actionable one, and a cleanup
 * that threw would replace it with a message about a delete request nobody asked for. The board
 * goes last and without `include_images`, so a generation that landed on the staging board while
 * the import ran survives as Uncategorized instead of being collected by a cleanup that never
 * meant it.
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

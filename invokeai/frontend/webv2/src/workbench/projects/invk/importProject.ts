import { mapWithConcurrency } from '@platform/core/concurrency';
import { collectLiveAssetRefs, selectCoverImageName } from '@workbench/projects/projectAssets';

import type { InvkBoardSnapshot } from './board';
import type {
  MediaMaterializer,
  RestoredMediaLedger,
  RestoreProjectMediaDeps,
  RestoreProjectMediaResult,
} from './restoreProjectMedia';
import type { InvkMediaRef } from './transfer';

import { INVK_MAX_ARCHIVE_BYTES, readArchive, readEntryText } from './archive';
import {
  INVK_TRANSFER_CONCURRENCY,
  isRequestCancellation,
  mimeForEntryName,
  uploadBoardImage,
  uploadBoardVideo,
} from './assetTransport';
import { parseInvkBoardSnapshot } from './board';
import {
  INVK_BOARD_ENTRY,
  INVK_DOCUMENT_ENTRY,
  INVK_IMAGES_PREFIX,
  INVK_MANIFEST_ENTRY,
  INVK_VIDEOS_PREFIX,
  InvkFormatError,
} from './format';
import { type InvkManifest, parseInvkManifest } from './manifest';
import { restoreProjectMedia } from './restoreProjectMedia';
import { toMediaRefs } from './transfer';

/**
 * Reading an `.invk` back, in two steps a caller can put a decision between.
 *
 * {@link readInvkArchive} is pure inspection — no network, no mutation — so everything structural
 * fails before a board or a single uploaded image exists. {@link restoreArchiveMedia} is the part
 * with consequences. Neither touches the document: the caller applies the returned mapping,
 * because the caller is also the one assigning the new project id.
 */

export interface InvkArchiveContents {
  /**
   * What the project's board held, or `null` for an archive that names no board. `null` and
   * `{items: []}` are different answers: only the first may have a board invented for it.
   */
  boardSnapshot: InvkBoardSnapshot | null;
  /** Bundled preview bytes and the entry they came from, when the archive has one. */
  cover: { bytes: Uint8Array; entryName: string } | null;
  /** Bundled image bytes, keyed by the image name the exporting server used. */
  images: Map<string, Uint8Array>;
  manifest: InvkManifest;
  projectDocument: Record<string, unknown>;
  /** Bundled video bytes, keyed by the video name the exporting server used. */
  videos: Map<string, Uint8Array>;
}

const parseDocumentEntry = (bytes: Uint8Array): Record<string, unknown> => {
  let parsed: unknown;

  try {
    parsed = JSON.parse(readEntryText(bytes));
  } catch {
    throw new InvkFormatError('damaged', `${INVK_DOCUMENT_ENTRY} is not valid JSON.`);
  }

  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new InvkFormatError('damaged', `${INVK_DOCUMENT_ENTRY} is not a project document.`);
  }

  return parsed as Record<string, unknown>;
};

/**
 * The board enumeration, or `null` when the archive carries none. Absent is valid; *malformed* is
 * not, and must fail before anything is created — half an enumeration is not a board.
 */
const parseBoardEntry = (entries: ReadonlyMap<string, Uint8Array>): InvkBoardSnapshot | null => {
  const entry = entries.get(INVK_BOARD_ENTRY);

  if (!entry) {
    return null;
  }

  try {
    return parseInvkBoardSnapshot(JSON.parse(readEntryText(entry)));
  } catch (error) {
    if (error instanceof InvkFormatError) {
      throw error;
    }

    throw new InvkFormatError('damaged', `${INVK_BOARD_ENTRY} is not valid JSON.`);
  }
};

/**
 * The part of `images/<name>` that is a name, or `null` otherwise. A ZIP path is attacker-controlled:
 * `images/../../x.png` would let an entry masquerade as media the board enumeration asks for by
 * name. Server media names never contain a separator, so the check costs nothing legitimate.
 */
const toSafeBasename = (path: string, prefix: string): string | null => {
  const name = path.slice(prefix.length);

  return name === '' || name === '.' || name === '..' || /[/\\\0]/u.test(name) ? null : name;
};

/** Unpack and validate an archive. Throws {@link InvkFormatError} and nothing else. */
export const readInvkArchive = async (file: File): Promise<InvkArchiveContents> => {
  if (file.size > INVK_MAX_ARCHIVE_BYTES) {
    throw new InvkFormatError('too-large', `Project archive is ${file.size} bytes.`);
  }

  const entries = await readArchive(new Uint8Array(await file.arrayBuffer()));
  const manifestEntry = entries.get(INVK_MANIFEST_ENTRY);

  if (!manifestEntry) {
    throw new InvkFormatError('not-a-project', `Archive has no ${INVK_MANIFEST_ENTRY}.`);
  }

  let manifestData: unknown;

  try {
    manifestData = JSON.parse(readEntryText(manifestEntry));
  } catch {
    throw new InvkFormatError('not-a-project', `${INVK_MANIFEST_ENTRY} is not valid JSON.`);
  }

  const manifest = parseInvkManifest(manifestData);
  const documentEntry = entries.get(INVK_DOCUMENT_ENTRY);

  if (!documentEntry) {
    throw new InvkFormatError('damaged', `Archive has no ${INVK_DOCUMENT_ENTRY}.`);
  }

  // Before the bytes are indexed: an archive whose enumeration is malformed must create nothing,
  // and the enumeration is what decides which of those bytes are board media at all.
  const boardSnapshot = parseBoardEntry(entries);
  const images = new Map<string, Uint8Array>();
  const videos = new Map<string, Uint8Array>();

  for (const [path, bytes] of entries) {
    for (const [prefix, store] of [
      [INVK_IMAGES_PREFIX, images],
      [INVK_VIDEOS_PREFIX, videos],
    ] as const) {
      if (!path.startsWith(prefix)) {
        continue;
      }

      const name = toSafeBasename(path, prefix);

      if (name !== null) {
        store.set(name, bytes);
      }
    }
  }

  const coverBytes = manifest.cover === undefined ? undefined : entries.get(manifest.cover);

  return {
    boardSnapshot,
    cover: coverBytes === undefined ? null : { bytes: coverBytes, entryName: manifest.cover! },
    images,
    manifest,
    projectDocument: parseDocumentEntry(documentEntry),
    videos,
  };
};

export interface ArchiveBoardUploadDeps {
  signal?: AbortSignal;
  uploadBoardImage?: typeof uploadBoardImage;
  uploadBoardVideo?: typeof uploadBoardVideo;
}

/**
 * The import half of the materialization seam: board media comes out of the archive. Uploaded with
 * no existence check — see `transfer.ts`. A descriptor the archive carries no bytes for is reported
 * rather than skipped: the exporting server said it was there.
 *
 * Entries are dropped as they land. An `.invk` may be two gigabytes, and holding every one until
 * the import finishes means holding the archive *and* everything unpacked from it.
 */
export const createArchiveMediaMaterializer = (
  archive: Pick<InvkArchiveContents, 'images' | 'videos'>,
  deps: ArchiveBoardUploadDeps = {}
): MediaMaterializer => {
  const uploadImage = deps.uploadBoardImage ?? uploadBoardImage;
  const uploadVideo = deps.uploadBoardVideo ?? uploadBoardVideo;

  return async (items, boardId, onItemSettled) => {
    const result: Awaited<ReturnType<MediaMaterializer>> = { failed: [], materialized: [] };

    await mapWithConcurrency(items, INVK_TRANSFER_CONCURRENCY, async (item) => {
      const entries = item.kind === 'image' ? archive.images : archive.videos;
      const bytes = entries.get(item.name);

      if (bytes === undefined) {
        result.failed.push({ kind: item.kind, name: item.name, reason: 'missing-entry' });
        onItemSettled();

        return;
      }

      try {
        const options = {
          boardId,
          category: item.category,
          contentType: mimeForEntryName(item.name, item.kind),
          ...(deps.signal === undefined ? {} : { signal: deps.signal }),
        };
        const name =
          item.kind === 'image'
            ? (await uploadImage(bytes, item.name, options)).imageName
            : (await uploadVideo(bytes, item.name, options)).videoName;

        result.materialized.push({ kind: item.kind, name, sourceName: item.name });
        entries.delete(item.name);
      } catch (error) {
        // A cancelled signal or an expired account is not this item failing — it is every
        // remaining item failing at once, which is an ended operation rather than a lossy one.
        if (isRequestCancellation(error)) {
          throw error;
        }

        result.failed.push({ kind: item.kind, name: item.name, reason: 'upload-failed' });
      }

      onItemSettled();
    });

    return result;
  };
};

export interface RestoreArchiveMediaInput {
  /** The staging board for this archive's board media, or `null` when it carries none. */
  boardId: string | null;
  /** The canonical document, already rehydrated and re-serialized. */
  projectDocument: Record<string, unknown>;
  /** The id the project will be created under. */
  projectId: string;
  /** Written as the restore runs, so a failure can undo exactly what it made. */
  ledger: RestoredMediaLedger;
}

export type RestoreArchiveMediaDeps = ArchiveBoardUploadDeps &
  Omit<RestoreProjectMediaDeps, 'documentMediaBytes' | 'materializeBoardMedia'>;

/**
 * Make an archive's media exist here: board items onto the staging board under fresh identities,
 * document-only references deduplicated against what is already present.
 */
export const restoreArchiveMedia = (
  archive: InvkArchiveContents,
  input: RestoreArchiveMediaInput,
  deps: RestoreArchiveMediaDeps = {}
): Promise<RestoreProjectMediaResult> => {
  const { signal, uploadBoardImage: overrideImage, uploadBoardVideo: overrideVideo, ...restoreDeps } = deps;
  const bytesFor = (ref: InvkMediaRef): Uint8Array | undefined =>
    (ref.kind === 'image' ? archive.images : archive.videos).get(ref.name);

  return restoreProjectMedia(
    {
      boardId: input.boardId,
      boardItems: archive.boardSnapshot?.items ?? [],
      coverBytes: archive.cover,
      coverSourceImageName: selectCoverImageName(input.projectDocument),
      documentRefs: toMediaRefs(collectLiveAssetRefs(input.projectDocument)),
      ledger: input.ledger,
      projectId: input.projectId,
    },
    {
      ...restoreDeps,
      documentMediaBytes: bytesFor,
      materializeBoardMedia: createArchiveMediaMaterializer(archive, {
        ...(signal === undefined ? {} : { signal }),
        ...(overrideImage === undefined ? {} : { uploadBoardImage: overrideImage }),
        ...(overrideVideo === undefined ? {} : { uploadBoardVideo: overrideVideo }),
      }),
      ...(signal === undefined ? {} : { signal }),
    }
  );
};

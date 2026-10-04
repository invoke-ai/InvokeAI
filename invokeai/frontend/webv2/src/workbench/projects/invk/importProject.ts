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
import { type EmbeddedFont, readEmbeddedFonts } from './fonts';
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

/** Inspect before creating resources; callers own document remapping and fresh project identity. */

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
  fonts?: EmbeddedFont[];
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

/** An absent board entry is valid; a malformed one must fail before mutations. */
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

/** Reject path separators so malicious ZIP paths cannot impersonate board media. */
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

  // Validate enumeration before indexing bytes or creating resources.
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
  const projectDocument = parseDocumentEntry(documentEntry);
  const fonts = await readEmbeddedFonts(manifest.fonts ?? [], projectDocument, entries);

  return {
    boardSnapshot,
    cover: coverBytes === undefined ? null : { bytes: coverBytes, entryName: manifest.cover! },
    images,
    fonts,
    manifest,
    projectDocument,
    videos,
  };
};

export interface ArchiveBoardUploadDeps {
  signal?: AbortSignal;
  uploadBoardImage?: typeof uploadBoardImage;
  uploadBoardVideo?: typeof uploadBoardVideo;
}

/** Upload fresh identities, report missing bytes, and release materialized entries to bound memory. */
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
        // Cancellation or account expiry aborts the operation rather than counting as per-item loss.
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

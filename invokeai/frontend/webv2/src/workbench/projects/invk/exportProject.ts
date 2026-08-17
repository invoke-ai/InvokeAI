import { mapWithConcurrency } from '@platform/core/concurrency';
import { collectLiveAssetRefs, selectCoverImageName, stripInstallationState } from '@workbench/projects/projectAssets';

import type { InvkArchiveEntry } from './archive';
import type { FetchedThumbnail } from './assetTransport';
import type { InvkBoardItem, InvkBoardSnapshot } from './board';
import type { InvkTransferItem, ProjectTransferIssues } from './transfer';

import { binaryEntry, INVK_MAX_ENTRIES, textEntry, writeArchive } from './archive';
import {
  coverExtensionForMime,
  createAssetExportTransport,
  INVK_TRANSFER_CONCURRENCY,
  isRequestCancellation,
} from './assetTransport';
import { buildInvkBoardSnapshot } from './board';
import {
  INVK_BOARD_ENTRY,
  INVK_DOCUMENT_ENTRY,
  INVK_IMAGES_PREFIX,
  INVK_MANIFEST_ENTRY,
  INVK_VIDEOS_PREFIX,
  InvkFormatError,
} from './format';
import { buildInvkManifest, toInvkFileName } from './manifest';
import { createTransferIssueLog, planMediaTransfer, toMediaRefs } from './transfer';

/**
 * Writing an `.invk`, split into a pure planner and an impure executor, as
 * `canvas-engine/export/psdExport.ts` splits PSD export. {@link planInvkExport} decides what the
 * archive contains from the document alone, so that decision is a node test.
 *
 * An unservable image is logged and skipped, never fatal. Cancellation is the one failure that is
 * *not* a skip: it makes every asset unservable at once, and skipping all of them would pack an
 * archive of nothing and hand it over as a finished download. The signal is checked once more
 * before the archive is written.
 */

export interface InvkExportPlan {
  /** The board's contents exactly as they will be written to `board.json`. */
  boardSnapshot: InvkBoardSnapshot;
  /** Cover image name, or `null` for a project that has produced nothing. */
  coverImageName: string | null;
  /** The project document, already serialized. */
  documentJson: string;
  /** Download file name, including the extension. */
  fileName: string;
  manifestInput: { appVersion: string; createdAt: string; name: string; sourceProjectId?: string };
  /** Board membership and document references merged, each item exactly once. */
  transferItems: InvkTransferItem[];
}

/**
 * Fixed entries every archive carries: manifest, document, board. The cover is counted separately
 * because it is optional.
 */
const FIXED_ENTRY_COUNT = 3;

export const planInvkExport = (input: {
  appVersion: string;
  boardItems: readonly InvkBoardItem[];
  createdAt: string;
  name: string;
  projectDocument: Record<string, unknown>;
}): InvkExportPlan => {
  const sourceProjectId = typeof input.projectDocument.id === 'string' ? input.projectDocument.id : undefined;

  // Installation state is dropped here rather than left to the collector's skip list: not bundling
  // a reference does not stop it travelling, it only stops it arriving with pixels. The planner is
  // where "what belongs in a project file" is decided, so it is where the answer is applied.
  const projectDocument = stripInstallationState(input.projectDocument);
  const boardSnapshot = buildInvkBoardSnapshot(input.boardItems);
  const transferItems = planMediaTransfer(boardSnapshot.items, toMediaRefs(collectLiveAssetRefs(projectDocument)));

  // Refused before a single byte is fetched. An archive that cannot be packed is not worth the
  // hundreds of round trips it would take to discover that at the end.
  const worstCaseEntries = FIXED_ENTRY_COUNT + transferItems.length + 1;

  if (worstCaseEntries > INVK_MAX_ENTRIES) {
    throw new InvkFormatError('too-large', `Project needs ${worstCaseEntries} archive entries`);
  }

  return {
    boardSnapshot,
    coverImageName: selectCoverImageName(projectDocument),
    // Compact rather than indented: the document is machine-read, and the two
    // bytes per line would be the largest entry in the archive before deflate.
    documentJson: JSON.stringify(projectDocument),
    fileName: toInvkFileName(input.name),
    manifestInput: {
      appVersion: input.appVersion,
      createdAt: input.createdAt,
      name: input.name,
      ...(sourceProjectId === undefined ? {} : { sourceProjectId }),
    },
    transferItems,
  };
};

export interface InvkExportProgress {
  completed: number;
  phase: 'bundling' | 'packing';
  total: number;
}

export interface InvkExportDeps {
  download: (blob: Blob, fileName: string) => void;
  fetchImageBytes?: (imageName: string, signal?: AbortSignal) => Promise<Uint8Array | null>;
  fetchImageThumbnail?: (imageName: string, signal?: AbortSignal) => Promise<FetchedThumbnail | null>;
  fetchVideoBytes?: (videoName: string, signal?: AbortSignal) => Promise<Uint8Array | null>;
  onProgress?: (progress: InvkExportProgress) => void;
  signal?: AbortSignal;
}

export interface InvkExportResult extends ProjectTransferIssues {
  /** Images successfully written into `images/`. */
  bundledImageCount: number;
  /** Videos successfully written into `videos/`. */
  bundledVideoCount: number;
}

export const executeInvkExport = async (plan: InvkExportPlan, deps: InvkExportDeps): Promise<InvkExportResult> => {
  const transport = createAssetExportTransport();
  const readImage = deps.fetchImageBytes ?? transport.fetchImageBytes;
  const readVideo = deps.fetchVideoBytes ?? transport.fetchVideoBytes;
  const readThumbnail = deps.fetchImageThumbnail ?? transport.fetchImageThumbnail;
  const entries = new Map<string, InvkArchiveEntry>();
  const issues = createTransferIssueLog();
  let bundledImageCount = 0;
  let bundledVideoCount = 0;
  let completed = 0;

  /** Unservable is `null`; cancelled rethrows. */
  const skipUnservable = async <T>(read: () => Promise<T | null>): Promise<T | null> => {
    try {
      return await read();
    } catch (error) {
      if (isRequestCancellation(error) || (error instanceof InvkFormatError && error.reason === 'too-large')) {
        throw error;
      }

      return null;
    }
  };

  const cover =
    plan.coverImageName === null ? null : await skipUnservable(() => readThumbnail(plan.coverImageName!, deps.signal));
  const coverEntryName = cover === null ? undefined : `cover.${coverExtensionForMime(cover.contentType)}`;

  // One queue over both kinds and both roles: an item that is on the board *and* referenced by the
  // document is one fetch, not two. The concurrency limit exists to be kind to the backend, and it
  // would not be if images and videos each got their own.
  const assets = plan.transferItems;

  await mapWithConcurrency(assets, INVK_TRANSFER_CONCURRENCY, async (item) => {
    const { kind, name } = item;
    const bytes = await skipUnservable(() =>
      kind === 'image' ? readImage(name, deps.signal) : readVideo(name, deps.signal)
    );

    completed += 1;
    deps.onProgress?.({ completed, phase: 'bundling', total: assets.length });

    if (bytes === null) {
      // Reported against every role it filled: the same failure costs a board result and a canvas
      // layer, and the person holding the file needs to know which.
      if (item.isBoardItem) {
        issues.addBoardItemIssue(item, 'fetch-failed');
      }

      if (item.isDocumentReference) {
        issues.addDocumentReferenceIssue(item, 'fetch-failed');
      }

      return;
    }

    if (kind === 'image') {
      bundledImageCount += 1;
      entries.set(`${INVK_IMAGES_PREFIX}${name}`, binaryEntry(bytes));

      return;
    }

    bundledVideoCount += 1;
    entries.set(`${INVK_VIDEOS_PREFIX}${name}`, binaryEntry(bytes));
  });

  // The last fetch may have completed just as the export stopped being wanted;
  // no fetch would have rejected in that window, and packing is the expensive
  // step that precedes handing a file to the browser.
  deps.signal?.throwIfAborted();

  deps.onProgress?.({ completed: assets.length, phase: 'packing', total: assets.length });

  const manifest = buildInvkManifest({
    ...plan.manifestInput,
    ...(coverEntryName === undefined ? {} : { cover: coverEntryName }),
  });

  // The manifest is the one entry a person may open by hand, so it is indented.
  entries.set(INVK_MANIFEST_ENTRY, textEntry(JSON.stringify(manifest, null, 2)));
  entries.set(INVK_DOCUMENT_ENTRY, textEntry(plan.documentJson));
  // Written even when empty. "This project's board held nothing" is a fact the reader needs; its
  // absence would be indistinguishable from a v2 archive that never knew about boards. The
  // descriptor is kept for an item whose bytes could not be fetched, so the loss is reported on
  // import rather than silently forgotten here.
  entries.set(INVK_BOARD_ENTRY, textEntry(JSON.stringify(plan.boardSnapshot)));

  if (cover !== null && coverEntryName !== undefined) {
    entries.set(coverEntryName, binaryEntry(cover.bytes));
  }

  const blob = await writeArchive(entries);

  deps.signal?.throwIfAborted();

  deps.download(blob, plan.fileName);

  return { bundledImageCount, bundledVideoCount, ...issues.toIssues() };
};

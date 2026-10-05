import { sha256Hex } from '@platform/browser/sha256';
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
  collectFontDependencies,
  INVK_MAX_FONT_BYTES,
  type FontArchiveTransport,
  type InvkFontDependency,
} from './fonts';
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

/** Skip unavailable assets, but abort on cancellation before packing. */

export interface InvkExportPlan {
  /** The board's contents exactly as they will be written to `board.json`. */
  boardSnapshot: InvkBoardSnapshot;
  /** Cover image name, or `null` for a project that has produced nothing. */
  coverImageName: string | null;
  /** The project document, already serialized. */
  documentJson: string;
  /** Download file name, including the extension. */
  fileName: string;
  fonts: InvkFontDependency[];
  includeFonts: boolean;
  manifestInput: {
    appVersion: string;
    createdAt: string;
    minimumCanvasSchemaVersion: number;
    name: string;
    sourceProjectId?: string;
  };
  /** Board membership and document references merged, each item exactly once. */
  transferItems: InvkTransferItem[];
}

/** Budget fixed entries separately from the optional cover. */
const FIXED_ENTRY_COUNT = 3;

export const planInvkExport = (input: {
  appVersion: string;
  boardItems: readonly InvkBoardItem[];
  createdAt: string;
  minimumCanvasSchemaVersion: number;
  name: string;
  projectDocument: Record<string, unknown>;
  includeFonts?: boolean;
}): InvkExportPlan => {
  const sourceProjectId = typeof input.projectDocument.id === 'string' ? input.projectDocument.id : undefined;

  // Strip installation state during planning; excluding its bytes alone leaves dangling references.
  const projectDocument = stripInstallationState(input.projectDocument);
  const boardSnapshot = buildInvkBoardSnapshot(input.boardItems);
  const transferItems = planMediaTransfer(boardSnapshot.items, toMediaRefs(collectLiveAssetRefs(projectDocument)));
  const fonts = collectFontDependencies(projectDocument);

  // Check the archive budget before downloading assets.
  const worstCaseEntries = FIXED_ENTRY_COUNT + transferItems.length + (input.includeFonts ? fonts.length : 0) + 1;

  if (worstCaseEntries > INVK_MAX_ENTRIES) {
    throw new InvkFormatError('too-large', `Project needs ${worstCaseEntries} archive entries`);
  }

  return {
    boardSnapshot,
    coverImageName: selectCoverImageName(projectDocument),
    // Serialize compactly to limit the uncompressed document size.
    documentJson: JSON.stringify(projectDocument),
    fileName: toInvkFileName(input.name),
    fonts,
    includeFonts: input.includeFonts ?? false,
    manifestInput: {
      appVersion: input.appVersion,
      createdAt: input.createdAt,
      minimumCanvasSchemaVersion: input.minimumCanvasSchemaVersion,
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
  fontTransport?: FontArchiveTransport;
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
  const fonts = plan.fonts.map((font) => ({ ...font }));
  const total = plan.transferItems.length + (plan.includeFonts ? fonts.length : 0);

  if (plan.includeFonts && fonts.length > 0) {
    const fontTransport = deps.fontTransport ?? (await import('./fontTransport')).createFontArchiveTransport();
    for (const font of fonts) {
      deps.signal?.throwIfAborted();
      const { bytes, filename } = await fontTransport.download(font, deps.signal);
      const extension = filename.split('.').at(-1)?.toLowerCase();
      if (
        !extension ||
        !['ttf', 'otf', 'woff', 'woff2'].includes(extension) ||
        bytes.byteLength > INVK_MAX_FONT_BYTES ||
        (await sha256Hex(bytes)) !== font.contentHash
      ) {
        throw new InvkFormatError('damaged', `The font file for ${font.label} is unavailable or changed.`);
      }
      font.entry = `fonts/${font.contentHash}.${extension}`;
      entries.set(font.entry, binaryEntry(bytes));
      completed += 1;
      deps.onProgress?.({ completed, phase: 'bundling', total });
    }
  }

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

  // Share one deduplicated queue and concurrency budget across media kinds and roles.
  const assets = plan.transferItems;

  await mapWithConcurrency(assets, INVK_TRANSFER_CONCURRENCY, async (item) => {
    const { kind, name } = item;
    const bytes = await skipUnservable(() =>
      kind === 'image' ? readImage(name, deps.signal) : readVideo(name, deps.signal)
    );

    completed += 1;
    deps.onProgress?.({ completed, phase: 'bundling', total });

    if (bytes === null) {
      // Report loss for each affected board/document role.
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

  // Check cancellation again before expensive packing.
  deps.signal?.throwIfAborted();

  deps.onProgress?.({ completed, phase: 'packing', total });

  const manifest = buildInvkManifest({
    ...plan.manifestInput,
    fonts,
    ...(coverEntryName === undefined ? {} : { cover: coverEntryName }),
  });

  // The manifest is the one entry a person may open by hand, so it is indented.
  entries.set(INVK_MANIFEST_ENTRY, textEntry(JSON.stringify(manifest, null, 2)));
  entries.set(INVK_DOCUMENT_ENTRY, textEntry(plan.documentJson));
  // Write an explicit empty board entry and retain missing-byte descriptors for import reporting.
  entries.set(INVK_BOARD_ENTRY, textEntry(JSON.stringify(plan.boardSnapshot)));

  if (cover !== null && coverEntryName !== undefined) {
    entries.set(coverEntryName, binaryEntry(cover.bytes));
  }

  const blob = await writeArchive(entries);

  deps.signal?.throwIfAborted();

  deps.download(blob, plan.fileName);

  return { bundledImageCount, bundledVideoCount, ...issues.toIssues() };
};

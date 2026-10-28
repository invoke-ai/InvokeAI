import { sha256Hex } from '@platform/browser/sha256';
import { mapWithConcurrency } from '@platform/core/concurrency';
import { collectLiveAssetRefs, selectCoverImageName, stripInstallationState } from '@workbench/projects/projectAssets';

import type { InvkArchiveEntry } from './archive';
import type { FetchedThumbnail } from './assetTransport';
import type { InvkBoard, InvkBoardSnapshot } from './board';
import type { InvkTransferItem, ProjectTransferIssues } from './transfer';

import { binaryEntry, INVK_MAX_ENTRIES, textEntry, writeArchive } from './archive';
import {
  coverExtensionForMime,
  createAssetExportTransport,
  INVK_TRANSFER_CONCURRENCY,
  isRequestCancellation,
} from './assetTransport';
import { buildInvkBoardSnapshot, flattenInvkBoardItems } from './board';
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
  INVK_WORKSPACE_ENTRY,
  InvkFormatError,
} from './format';
import { buildInvkManifest, toInvkFileName } from './manifest';
import { createTransferIssueLog, planMediaTransfer, toMediaRefs } from './transfer';

/** Skip unavailable assets, but abort on cancellation before packing. */

export interface InvkExportPlan {
  /** The project's boards exactly as they will be written to `board.json`. */
  boardSnapshot: InvkBoardSnapshot;
  /** Cover image name, or `null` for a project that has produced nothing. */
  coverImageName: string | null;
  /** `project.json`, already serialized: the project content, or a whole document an exporter could not split. */
  documentJson: string;
  /** `workspace.json`, already serialized, or `null` when the exporter's workspace stays out of the file. */
  workspaceJson: string | null;
  /** Download file name, including the extension. */
  fileName: string;
  fonts: InvkFontDependency[];
  includeFonts: boolean;
  manifestInput: {
    appVersion: string;
    createdAt: string;
    documentSchemaVersion: number;
    minimumCanvasSchemaVersion: number;
    name: string;
    sourceProjectId?: string;
  };
  /** Board membership and document references merged, each item exactly once. */
  transferItems: InvkTransferItem[];
}

/** Budget fixed entries separately from the optional cover. */
const FIXED_ENTRY_COUNT = 4;

export const planInvkExport = (input: {
  appVersion: string;
  /** The project's boards as the server enumerated them, the inbox among them. */
  boards: readonly InvkBoard[];
  createdAt: string;
  minimumCanvasSchemaVersion: number;
  name: string;
  /** What `project.json` carries, and the document schema it is written in. */
  content: Record<string, unknown>;
  documentSchemaVersion: number;
  /** The exporter's own workspace, or `null` to leave it out of the file. */
  workspace: Record<string, unknown> | null;
  includeFonts?: boolean;
}): InvkExportPlan => {
  const sourceProjectId = typeof input.content.id === 'string' ? input.content.id : undefined;

  // Strip installation state during planning; excluding its bytes alone leaves dangling references.
  const content = stripInstallationState(input.content);
  const workspace = input.workspace === null ? null : stripInstallationState(input.workspace);
  const written = workspace === null ? [content] : [content, workspace];
  // Only what the file carries: media a left-out workspace names (staged results, gallery recents) stays behind.
  const documentRefs = written.map(collectLiveAssetRefs).reduce((all, refs) => ({
    images: new Set([...all.images, ...refs.images]),
    videos: new Set([...all.videos, ...refs.videos]),
  }));
  const boardSnapshot = buildInvkBoardSnapshot(input.boards);
  const transferItems = planMediaTransfer(flattenInvkBoardItems(boardSnapshot), toMediaRefs(documentRefs));
  // A reader checks fonts against the canvas of each half it reads, the workspace's staging area included.
  const fonts = collectFontDependencies({ canvas: written.map((half) => half.canvas) });

  // Check the archive budget before downloading assets.
  const worstCaseEntries = FIXED_ENTRY_COUNT + transferItems.length + (input.includeFonts ? fonts.length : 0) + 1;

  if (worstCaseEntries > INVK_MAX_ENTRIES) {
    throw new InvkFormatError('too-large', `Project needs ${worstCaseEntries} archive entries`);
  }

  return {
    boardSnapshot,
    // The gallery's newest result is the workspace's, so without it the cover comes from the canvas.
    coverImageName: selectCoverImageName({ ...workspace, ...content }),
    // Serialize compactly to limit the uncompressed document size.
    documentJson: JSON.stringify(content),
    fileName: toInvkFileName(input.name),
    fonts,
    includeFonts: input.includeFonts ?? false,
    manifestInput: {
      appVersion: input.appVersion,
      createdAt: input.createdAt,
      documentSchemaVersion: input.documentSchemaVersion,
      minimumCanvasSchemaVersion: input.minimumCanvasSchemaVersion,
      name: input.name,
      ...(sourceProjectId === undefined ? {} : { sourceProjectId }),
    },
    transferItems,
    workspaceJson: workspace === null ? null : JSON.stringify(workspace),
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
  if (plan.workspaceJson !== null) {
    entries.set(INVK_WORKSPACE_ENTRY, textEntry(plan.workspaceJson));
  }
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

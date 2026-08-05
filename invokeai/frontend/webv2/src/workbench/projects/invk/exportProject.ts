import { mapWithConcurrency } from '@platform/core/concurrency';
import { collectLiveAssetRefs, selectCoverImageName } from '@workbench/projects/projectAssets';

import type { InvkArchiveEntry } from './archive';
import type { FetchedThumbnail } from './assetTransport';

import { binaryEntry, textEntry, writeArchive } from './archive';
import { coverExtensionForMime, fetchImageBytes, fetchImageThumbnail, fetchVideoBytes } from './assetTransport';
import { INVK_DOCUMENT_ENTRY, INVK_IMAGES_PREFIX, INVK_MANIFEST_ENTRY, INVK_VIDEOS_PREFIX } from './format';
import { buildInvkManifest, toInvkFileName } from './manifest';

/**
 * Writing an `.invk`, split into a pure planner and an impure executor the same
 * way `canvas-engine/export/psdExport.ts` splits PSD export.
 *
 * {@link planInvkExport} decides what the archive contains — entry names, which
 * asset names to bundle and of which kind, which image becomes the cover — from
 * the document alone. No network, no DOM, no fflate, so the interesting decision
 * (what belongs in a project file) is a node test rather than a manual round trip.
 *
 * {@link executeInvkExport} does the parts that can fail: fetching each image,
 * packing the ZIP, handing it to the browser.
 *
 * An image the server will not serve is logged in the result and skipped, never
 * fatal. Half a project's pixels beats none, and the reference survives in the
 * document either way — importing onto the machine that still has the image
 * resolves it.
 */

/** Simultaneous asset fetches. Matches the previous frontend's limit. */
const ASSET_FETCH_CONCURRENCY = 5;

export interface InvkExportPlan {
  /** Cover image name, or `null` for a project that has produced nothing. */
  coverImageName: string | null;
  /** The project document, already serialized. */
  documentJson: string;
  /** Download file name, including the extension. */
  fileName: string;
  /** Image names to bundle, in a stable order. */
  imageNames: string[];
  manifestInput: { appVersion: string; createdAt: string; name: string; sourceProjectId?: string };
  /** Video names to bundle, in a stable order. */
  videoNames: string[];
}

export const planInvkExport = (input: {
  appVersion: string;
  createdAt: string;
  name: string;
  projectDocument: Record<string, unknown>;
}): InvkExportPlan => {
  const sourceProjectId = typeof input.projectDocument.id === 'string' ? input.projectDocument.id : undefined;

  const refs = collectLiveAssetRefs(input.projectDocument);

  return {
    coverImageName: selectCoverImageName(input.projectDocument),
    // Compact rather than indented: the document is machine-read, and the two
    // bytes per line would be the largest entry in the archive before deflate.
    documentJson: JSON.stringify(input.projectDocument),
    fileName: toInvkFileName(input.name),
    imageNames: [...refs.images].sort(),
    manifestInput: {
      appVersion: input.appVersion,
      createdAt: input.createdAt,
      name: input.name,
      ...(sourceProjectId === undefined ? {} : { sourceProjectId }),
    },
    videoNames: [...refs.videos].sort(),
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

export interface InvkExportResult {
  /** Images successfully written into `images/`. */
  bundledImageCount: number;
  /** Videos successfully written into `videos/`. */
  bundledVideoCount: number;
  /** Referenced assets the server would not serve. Their references still ship. */
  missingAssetNames: string[];
}

export const executeInvkExport = async (plan: InvkExportPlan, deps: InvkExportDeps): Promise<InvkExportResult> => {
  const readImage = deps.fetchImageBytes ?? fetchImageBytes;
  const readVideo = deps.fetchVideoBytes ?? fetchVideoBytes;
  const readThumbnail = deps.fetchImageThumbnail ?? fetchImageThumbnail;
  const entries = new Map<string, InvkArchiveEntry>();
  const missingAssetNames: string[] = [];
  let bundledImageCount = 0;
  let bundledVideoCount = 0;
  let completed = 0;

  const cover =
    plan.coverImageName === null ? null : await readThumbnail(plan.coverImageName, deps.signal).catch(() => null);
  const coverEntryName = cover === null ? undefined : `cover.${coverExtensionForMime(cover.contentType)}`;

  // One queue over both kinds: the concurrency limit is there to be kind to the
  // backend, and it would not be if images and videos each got their own.
  const assets = [
    ...plan.imageNames.map((name) => ({ kind: 'image' as const, name })),
    ...plan.videoNames.map((name) => ({ kind: 'video' as const, name })),
  ];

  await mapWithConcurrency(assets, ASSET_FETCH_CONCURRENCY, async ({ kind, name }) => {
    const bytes = await (kind === 'image' ? readImage(name, deps.signal) : readVideo(name, deps.signal)).catch(
      () => null
    );

    completed += 1;
    deps.onProgress?.({ completed, phase: 'bundling', total: assets.length });

    if (bytes === null) {
      missingAssetNames.push(name);

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

  deps.onProgress?.({ completed: assets.length, phase: 'packing', total: assets.length });

  const manifest = buildInvkManifest({
    ...plan.manifestInput,
    ...(coverEntryName === undefined ? {} : { cover: coverEntryName }),
  });

  // The manifest is the one entry a person may open by hand, so it is indented.
  entries.set(INVK_MANIFEST_ENTRY, textEntry(JSON.stringify(manifest, null, 2)));
  entries.set(INVK_DOCUMENT_ENTRY, textEntry(plan.documentJson));

  if (cover !== null && coverEntryName !== undefined) {
    entries.set(coverEntryName, binaryEntry(cover.bytes));
  }

  const blob = await writeArchive(entries);

  deps.download(blob, plan.fileName);

  return { bundledImageCount, bundledVideoCount, missingAssetNames: missingAssetNames.sort() };
};

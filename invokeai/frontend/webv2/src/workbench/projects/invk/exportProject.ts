import { mapWithConcurrency } from '@platform/core/concurrency';
import { collectLiveImageRefs, selectCoverImageName } from '@workbench/projects/projectAssets';

import type { InvkArchiveEntry } from './archive';
import type { FetchedThumbnail } from './imageTransport';

import { binaryEntry, textEntry, writeArchive } from './archive';
import { INVK_DOCUMENT_ENTRY, INVK_IMAGES_PREFIX, INVK_MANIFEST_ENTRY } from './format';
import { coverExtensionForMime, fetchImageBytes, fetchImageThumbnail } from './imageTransport';
import { buildInvkManifest, toInvkFileName } from './manifest';

/**
 * Writing an `.invk`, split into a pure planner and an impure executor the same
 * way `canvas-engine/export/psdExport.ts` splits PSD export.
 *
 * {@link planInvkExport} decides what the archive contains — entry names, which
 * image names to bundle, which one becomes the cover — from the document alone.
 * No network, no DOM, no fflate, so the interesting decision (what belongs in a
 * project file) is a node test rather than a manual round trip.
 *
 * {@link executeInvkExport} does the parts that can fail: fetching each image,
 * packing the ZIP, handing it to the browser.
 *
 * An image the server will not serve is logged in the result and skipped, never
 * fatal. Half a project's pixels beats none, and the reference survives in the
 * document either way — importing onto the machine that still has the image
 * resolves it.
 */

/** Simultaneous image fetches. Matches the previous frontend's limit. */
const IMAGE_FETCH_CONCURRENCY = 5;

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
}

export const planInvkExport = (input: {
  appVersion: string;
  createdAt: string;
  name: string;
  projectDocument: Record<string, unknown>;
}): InvkExportPlan => {
  const sourceProjectId = typeof input.projectDocument.id === 'string' ? input.projectDocument.id : undefined;

  return {
    coverImageName: selectCoverImageName(input.projectDocument),
    // Compact rather than indented: the document is machine-read, and the two
    // bytes per line would be the largest entry in the archive before deflate.
    documentJson: JSON.stringify(input.projectDocument),
    fileName: toInvkFileName(input.name),
    imageNames: [...collectLiveImageRefs(input.projectDocument)].sort(),
    manifestInput: {
      appVersion: input.appVersion,
      createdAt: input.createdAt,
      name: input.name,
      ...(sourceProjectId === undefined ? {} : { sourceProjectId }),
    },
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
  onProgress?: (progress: InvkExportProgress) => void;
  signal?: AbortSignal;
}

export interface InvkExportResult {
  /** Images successfully written into `images/`. */
  bundledCount: number;
  /** Referenced images the server would not serve. Their references still ship. */
  missingImageNames: string[];
}

export const executeInvkExport = async (plan: InvkExportPlan, deps: InvkExportDeps): Promise<InvkExportResult> => {
  const readImage = deps.fetchImageBytes ?? fetchImageBytes;
  const readThumbnail = deps.fetchImageThumbnail ?? fetchImageThumbnail;
  const entries = new Map<string, InvkArchiveEntry>();
  const missingImageNames: string[] = [];
  let bundledCount = 0;
  let completed = 0;

  const cover =
    plan.coverImageName === null ? null : await readThumbnail(plan.coverImageName, deps.signal).catch(() => null);
  const coverEntryName = cover === null ? undefined : `cover.${coverExtensionForMime(cover.contentType)}`;

  await mapWithConcurrency(plan.imageNames, IMAGE_FETCH_CONCURRENCY, async (imageName) => {
    const bytes = await readImage(imageName, deps.signal).catch(() => null);

    completed += 1;
    deps.onProgress?.({ completed, phase: 'bundling', total: plan.imageNames.length });

    if (bytes === null) {
      missingImageNames.push(imageName);

      return;
    }

    bundledCount += 1;
    entries.set(`${INVK_IMAGES_PREFIX}${imageName}`, binaryEntry(bytes));
  });

  deps.onProgress?.({ completed: plan.imageNames.length, phase: 'packing', total: plan.imageNames.length });

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

  return { bundledCount, missingImageNames: missingImageNames.sort() };
};

import { mapWithConcurrency } from '@platform/core/concurrency';
import { collectLiveImageRefs } from '@workbench/projects/projectAssets';

import type { UploadedImage } from './imageTransport';
import type { InvkManifest } from './manifest';

import { readArchive, readEntryText } from './archive';
import { INVK_DOCUMENT_ENTRY, INVK_IMAGES_PREFIX, INVK_MANIFEST_ENTRY, InvkFormatError } from './format';
import { findExistingImageNames, mimeForEntryName, uploadArchiveImage } from './imageTransport';
import { parseInvkManifest } from './manifest';

/**
 * Reading an `.invk` back, in two steps a caller can put a decision between.
 *
 * {@link readInvkArchive} is pure inspection: unpack, validate the manifest,
 * parse the document, index the bundled bytes. It touches no network and mutates
 * nothing, so a caller can read a file, discover it is a legacy canvas project,
 * and say so without having created anything.
 *
 * {@link restoreArchiveImages} is the part with consequences: ask the server
 * which referenced images it already has, upload the rest out of the archive,
 * and report the old-to-new renames. It deliberately does *not* touch the
 * document — the caller applies the mapping, because the caller is also the one
 * assigning the new project id.
 *
 * Deduplication is what makes re-importing cheap and makes importing onto the
 * machine that exported it nearly free: an image already on the server is never
 * uploaded twice, so nothing accumulates.
 */

/** Simultaneous uploads. Matches the previous frontend's limit. */
const IMAGE_UPLOAD_CONCURRENCY = 5;

export interface InvkArchiveContents {
  /** Bundled preview bytes and the entry they came from, when the archive has one. */
  cover: { bytes: Uint8Array; entryName: string } | null;
  /** Bundled image bytes, keyed by the image name the exporting server used. */
  images: Map<string, Uint8Array>;
  manifest: InvkManifest;
  projectDocument: Record<string, unknown>;
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

/** Unpack and validate an archive. Throws {@link InvkFormatError} and nothing else. */
export const readInvkArchive = async (file: File): Promise<InvkArchiveContents> => {
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

  const images = new Map<string, Uint8Array>();

  for (const [path, bytes] of entries) {
    if (path.startsWith(INVK_IMAGES_PREFIX)) {
      images.set(path.slice(INVK_IMAGES_PREFIX.length), bytes);
    }
  }

  const coverBytes = manifest.cover === undefined ? undefined : entries.get(manifest.cover);

  return {
    cover: coverBytes === undefined ? null : { bytes: coverBytes, entryName: manifest.cover! },
    images,
    manifest,
    projectDocument: parseDocumentEntry(documentEntry),
  };
};

export interface RestoreImagesResult {
  /** Referenced images the archive did not carry and the server does not have. */
  danglingImageNames: string[];
  /** Old name to new name, for the images the server renamed on upload. */
  mapping: Map<string, string>;
  /** The uploaded cover's server name, when the archive carried one. */
  coverImageName: string | null;
  /** Images uploaded from the archive. */
  uploadedCount: number;
}

export interface RestoreImagesDeps {
  findExistingImageNames?: (imageNames: readonly string[], signal?: AbortSignal) => Promise<Set<string>>;
  onProgress?: (progress: { completed: number; total: number }) => void;
  signal?: AbortSignal;
  uploadArchiveImage?: (
    bytes: Uint8Array,
    fileName: string,
    options?: { contentType?: string; signal?: AbortSignal }
  ) => Promise<UploadedImage>;
}

/**
 * Make the archive's images available on this server. Only what is missing gets
 * uploaded; a referenced image that is neither on the server nor in the archive
 * is reported as dangling and its reference is left exactly as it is, so the
 * project still opens with one broken layer rather than not at all.
 */
export const restoreArchiveImages = async (
  contents: InvkArchiveContents,
  deps: RestoreImagesDeps = {}
): Promise<RestoreImagesResult> => {
  const checkExisting = deps.findExistingImageNames ?? findExistingImageNames;
  const upload = deps.uploadArchiveImage ?? uploadArchiveImage;
  const referenced = [...collectLiveImageRefs(contents.projectDocument)].sort();
  const existing = await checkExisting(referenced, deps.signal);
  const missing = referenced.filter((imageName) => !existing.has(imageName));
  const uploadable = missing.filter((imageName) => contents.images.has(imageName));
  const danglingImageNames = missing.filter((imageName) => !contents.images.has(imageName));
  const mapping = new Map<string, string>();

  // The cover is not part of the document, so it is uploaded alongside rather
  // than through the reference set.
  const coverTotal = contents.cover === null ? 0 : 1;
  let completed = 0;
  let uploadedCount = 0;

  const coverImageName =
    contents.cover === null
      ? null
      : await upload(contents.cover.bytes, contents.cover.entryName, {
          contentType: mimeForEntryName(contents.cover.entryName),
          signal: deps.signal,
        })
          .then((uploaded) => uploaded.imageName)
          .catch(() => null);

  if (contents.cover !== null) {
    completed += 1;
    deps.onProgress?.({ completed, total: uploadable.length + coverTotal });
  }

  await mapWithConcurrency(uploadable, IMAGE_UPLOAD_CONCURRENCY, async (imageName) => {
    const bytes = contents.images.get(imageName)!;

    try {
      const uploaded = await upload(bytes, imageName, {
        contentType: mimeForEntryName(imageName),
        signal: deps.signal,
      });

      uploadedCount += 1;

      if (uploaded.imageName !== imageName) {
        mapping.set(imageName, uploaded.imageName);
      }
    } catch {
      // A failed upload leaves the reference pointing at a name this server does
      // not have — the same outcome as an image the archive never carried.
      danglingImageNames.push(imageName);
    }

    completed += 1;
    deps.onProgress?.({ completed, total: uploadable.length + coverTotal });
  });

  return { coverImageName, danglingImageNames: danglingImageNames.sort(), mapping, uploadedCount };
};

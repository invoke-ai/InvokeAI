import { mapWithConcurrency } from '@platform/core/concurrency';
import { collectLiveAssetRefs } from '@workbench/projects/projectAssets';

import type { UploadedImage, UploadedVideo } from './assetTransport';
import type { InvkManifest } from './manifest';

import { readArchive, readEntryText } from './archive';
import {
  findExistingImageNames,
  findExistingVideoNames,
  mimeForEntryName,
  uploadArchiveImage,
  uploadArchiveVideo,
} from './assetTransport';
import {
  INVK_DOCUMENT_ENTRY,
  INVK_IMAGES_PREFIX,
  INVK_MANIFEST_ENTRY,
  INVK_VIDEOS_PREFIX,
  InvkFormatError,
} from './format';
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
const ASSET_UPLOAD_CONCURRENCY = 5;

export interface InvkArchiveContents {
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
  const videos = new Map<string, Uint8Array>();

  for (const [path, bytes] of entries) {
    if (path.startsWith(INVK_IMAGES_PREFIX)) {
      images.set(path.slice(INVK_IMAGES_PREFIX.length), bytes);
      continue;
    }

    if (path.startsWith(INVK_VIDEOS_PREFIX)) {
      videos.set(path.slice(INVK_VIDEOS_PREFIX.length), bytes);
    }
  }

  const coverBytes = manifest.cover === undefined ? undefined : entries.get(manifest.cover);

  return {
    cover: coverBytes === undefined ? null : { bytes: coverBytes, entryName: manifest.cover! },
    images,
    manifest,
    projectDocument: parseDocumentEntry(documentEntry),
    videos,
  };
};

export interface RestoreAssetsResult {
  /** The uploaded cover's server name, when the archive carried one. */
  coverImageName: string | null;
  /** Referenced assets the archive did not carry and the server does not have, of either kind. */
  danglingAssetNames: string[];
  /** Old name to new name, per kind. Kept apart because the namespaces are. */
  mappings: { images: Map<string, string>; videos: Map<string, string> };
  /** Assets uploaded from the archive, of either kind. */
  uploadedCount: number;
}

export interface RestoreAssetsDeps {
  findExistingImageNames?: (imageNames: readonly string[], signal?: AbortSignal) => Promise<Set<string>>;
  findExistingVideoNames?: (videoNames: readonly string[], signal?: AbortSignal) => Promise<Set<string>>;
  onProgress?: (progress: { completed: number; total: number }) => void;
  signal?: AbortSignal;
  uploadArchiveImage?: (
    bytes: Uint8Array,
    fileName: string,
    options?: { contentType?: string; signal?: AbortSignal }
  ) => Promise<UploadedImage>;
  uploadArchiveVideo?: (
    bytes: Uint8Array,
    fileName: string,
    options?: { contentType?: string; signal?: AbortSignal }
  ) => Promise<UploadedVideo>;
}

interface PendingUpload {
  bytes: Uint8Array;
  kind: 'image' | 'video';
  name: string;
}

/**
 * Make the archive's assets available on this server. Only what is missing gets
 * uploaded; a referenced asset that is neither on the server nor in the archive
 * is reported as dangling and its reference is left exactly as it is, so the
 * project still opens with one broken layer rather than not at all.
 */
export const restoreArchiveAssets = async (
  contents: InvkArchiveContents,
  deps: RestoreAssetsDeps = {}
): Promise<RestoreAssetsResult> => {
  const checkExistingImages = deps.findExistingImageNames ?? findExistingImageNames;
  const checkExistingVideos = deps.findExistingVideoNames ?? findExistingVideoNames;
  const uploadImage = deps.uploadArchiveImage ?? uploadArchiveImage;
  const uploadVideo = deps.uploadArchiveVideo ?? uploadArchiveVideo;

  const referenced = collectLiveAssetRefs(contents.projectDocument);
  const referencedImages = [...referenced.images].sort();
  const referencedVideos = [...referenced.videos].sort();

  // Both existence checks go out before either upload does; they are
  // independent, and a project with videos should not wait out the image pass.
  const [existingImages, existingVideos] = await Promise.all([
    checkExistingImages(referencedImages, deps.signal),
    referencedVideos.length === 0
      ? Promise.resolve(new Set<string>())
      : checkExistingVideos(referencedVideos, deps.signal),
  ]);

  const danglingAssetNames: string[] = [];
  const pending: PendingUpload[] = [];

  for (const [names, existing, store, kind] of [
    [referencedImages, existingImages, contents.images, 'image'],
    [referencedVideos, existingVideos, contents.videos, 'video'],
  ] as const) {
    for (const name of names) {
      if (existing.has(name)) {
        continue;
      }

      const bytes = store.get(name);

      if (bytes === undefined) {
        danglingAssetNames.push(name);
        continue;
      }

      pending.push({ bytes, kind, name });
    }
  }

  const mappings = { images: new Map<string, string>(), videos: new Map<string, string>() };

  // The cover is not part of the document, so it is uploaded alongside rather
  // than through the reference set.
  const coverTotal = contents.cover === null ? 0 : 1;
  const total = pending.length + coverTotal;
  let completed = 0;
  let uploadedCount = 0;

  const coverImageName =
    contents.cover === null
      ? null
      : await uploadImage(contents.cover.bytes, contents.cover.entryName, {
          contentType: mimeForEntryName(contents.cover.entryName),
          signal: deps.signal,
        })
          .then((uploaded) => uploaded.imageName)
          .catch(() => null);

  if (contents.cover !== null) {
    completed += 1;
    deps.onProgress?.({ completed, total });
  }

  await mapWithConcurrency(pending, ASSET_UPLOAD_CONCURRENCY, async ({ bytes, kind, name }) => {
    try {
      const options = { contentType: mimeForEntryName(name), signal: deps.signal };
      const restoredName =
        kind === 'image'
          ? (await uploadImage(bytes, name, options)).imageName
          : (await uploadVideo(bytes, name, options)).videoName;

      uploadedCount += 1;

      if (restoredName !== name) {
        mappings[kind === 'image' ? 'images' : 'videos'].set(name, restoredName);
      }
    } catch {
      // A failed upload leaves the reference pointing at a name this server does
      // not have — the same outcome as an asset the archive never carried.
      danglingAssetNames.push(name);
    }

    completed += 1;
    deps.onProgress?.({ completed, total });
  });

  return { coverImageName, danglingAssetNames: danglingAssetNames.sort(), mappings, uploadedCount };
};

import { mapWithConcurrency } from '@platform/core/concurrency';
import { collectLiveAssetRefs, selectCoverImageName } from '@workbench/projects/projectAssets';

import type { UploadedImage, UploadedVideo } from './assetTransport';
import type { InvkManifest } from './manifest';
import type { InvkMediaIssue } from './transfer';

import { INVK_MAX_ARCHIVE_BYTES, readArchive, readEntryText } from './archive';
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
import { createTransferIssueLog } from './transfer';

/**
 * Reading an `.invk` back, in two steps a caller can put a decision between.
 *
 * {@link readInvkArchive} is pure inspection: unpack, validate the manifest,
 * parse the document, index the bundled bytes. It touches no network and mutates
 * nothing, so a caller can read a file, discover it is a legacy canvas project,
 * and say so without having created anything.
 *
 * {@link restoreArchiveAssets} is the part with consequences: ask the server
 * which referenced assets it already has, upload the rest out of the archive,
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
  /** References neither the archive nor this server could satisfy, with the reason. */
  documentReferenceIssues: InvkMediaIssue[];
  /** Old name to new name, per kind. Kept apart because the namespaces are. */
  mappings: { images: Map<string, string>; videos: Map<string, string> };
  /** Authoritative server identities created by this restore, for failure rollback. */
  uploadedAssets: UploadedArchiveAssets;
  /** Assets uploaded from the archive, of either kind. */
  uploadedCount: number;
}

export interface UploadedArchiveAssets {
  imageNames: string[];
  videoNames: string[];
}

export interface RollbackArchiveAssetsDeps {
  deleteArchiveImages?: (imageNames: string[], signal?: AbortSignal) => Promise<void>;
  deleteArchiveVideos?: (videoNames: string[], signal?: AbortSignal) => Promise<void>;
  signal?: AbortSignal;
}

/** Best-effort cleanup for server identities created by an import that could not create its project. */
export const rollbackArchiveAssets = async (
  uploadedAssets: UploadedArchiveAssets,
  deps: RollbackArchiveAssetsDeps = {}
): Promise<void> => {
  const transport = await import('./assetTransport');
  const deleteImages = deps.deleteArchiveImages ?? transport.deleteArchiveImages;
  const deleteVideos = deps.deleteArchiveVideos ?? transport.deleteArchiveVideos;
  const deletions: Promise<void>[] = [];

  if (uploadedAssets.imageNames.length > 0) {
    deletions.push(deleteImages(uploadedAssets.imageNames, deps.signal));
  }

  if (uploadedAssets.videoNames.length > 0) {
    deletions.push(deleteVideos(uploadedAssets.videoNames, deps.signal));
  }

  await Promise.allSettled(deletions);
};

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
 * The cover, preferring an image the restore already put on this server.
 *
 * The archive's `cover` entry is a thumbnail of an image the document also
 * references, and `getProjectCoverUrl` asks for a thumbnail anyway — so once
 * that image is restored, pointing the index at it is the same picture without
 * a second copy. Uploading the entry unconditionally instead left one orphan
 * per import: covers go up under the canvas's private `'other'` category, which
 * appears in no gallery view and no board count, so nobody could ever find or
 * delete them.
 *
 * The bundled bytes are the fallback, for the case that made them worth
 * carrying: a cover whose source image is dangling here.
 */
const resolveCoverImageName = async (input: {
  contents: InvkArchiveContents;
  mappings: { images: Map<string, string> };
  newlyUploadedImageNames: Set<string>;
  restoredImageNames: ReadonlySet<string>;
  signal?: AbortSignal;
  uploadImage: NonNullable<RestoreAssetsDeps['uploadArchiveImage']>;
}): Promise<string | null> => {
  const documentCoverName = selectCoverImageName(input.contents.projectDocument);

  if (documentCoverName !== null) {
    const restoredName = input.mappings.images.get(documentCoverName) ?? documentCoverName;

    if (input.restoredImageNames.has(restoredName)) {
      return restoredName;
    }
  }

  const { cover } = input.contents;

  if (cover === null) {
    return null;
  }

  try {
    const uploaded = await input.uploadImage(cover.bytes, cover.entryName, {
      contentType: mimeForEntryName(cover.entryName),
      signal: input.signal,
    });

    input.newlyUploadedImageNames.add(uploaded.imageName);

    return uploaded.imageName;
  } catch {
    // A project with no cover shows the folder glyph, which is a state the
    // library already renders. Not worth failing an import over.
    return null;
  }
};

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

  const issues = createTransferIssueLog();
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
        issues.addDocumentReferenceIssue({ kind, name }, 'missing-entry');
        continue;
      }

      pending.push({ bytes, kind, name });
    }
  }

  const mappings = { images: new Map<string, string>(), videos: new Map<string, string>() };

  // The cover is not part of the document, so it does not come through the
  // reference set; it is still counted here so progress spans the whole restore.
  const total = pending.length + (contents.cover === null ? 0 : 1);
  let completed = 0;
  let uploadedCount = 0;
  const newlyUploadedImageNames = new Set<string>();
  const newlyUploadedVideoNames = new Set<string>();

  /** Images this server can serve when the restore is done, under their final names. */
  const restoredImageNames = new Set(referencedImages.filter((name) => existingImages.has(name)));

  await mapWithConcurrency(pending, ASSET_UPLOAD_CONCURRENCY, async ({ bytes, kind, name }) => {
    try {
      const options = { contentType: mimeForEntryName(name, kind), signal: deps.signal };
      const restoredName =
        kind === 'image'
          ? (await uploadImage(bytes, name, options)).imageName
          : (await uploadVideo(bytes, name, options)).videoName;

      uploadedCount += 1;

      if (kind === 'image') {
        newlyUploadedImageNames.add(restoredName);
        restoredImageNames.add(restoredName);
      } else {
        newlyUploadedVideoNames.add(restoredName);
      }

      if (restoredName !== name) {
        mappings[kind === 'image' ? 'images' : 'videos'].set(name, restoredName);
      }
    } catch {
      // A failed upload leaves the reference pointing at a name this server does
      // not have — the same outcome as an asset the archive never carried.
      issues.addDocumentReferenceIssue({ kind, name }, 'upload-failed');
    }

    completed += 1;
    deps.onProgress?.({ completed, total });
  });

  const coverImageName = await resolveCoverImageName({
    contents,
    mappings,
    newlyUploadedImageNames,
    restoredImageNames,
    signal: deps.signal,
    uploadImage,
  });

  if (contents.cover !== null) {
    completed += 1;
    deps.onProgress?.({ completed, total });
  }

  return {
    coverImageName,
    documentReferenceIssues: issues.toIssues().documentReferenceIssues,
    mappings,
    uploadedAssets: {
      imageNames: [...newlyUploadedImageNames].sort(),
      videoNames: [...newlyUploadedVideoNames].sort(),
    },
    uploadedCount,
  };
};

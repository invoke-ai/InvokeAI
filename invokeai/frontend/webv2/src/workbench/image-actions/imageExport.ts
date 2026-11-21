import type { GalleryImage } from '@features/gallery';
import type { ImageExportSettings } from '@features/gallery/contracts';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { QueryClient } from '@tanstack/react-query';

import { getImageExportEncoding, normalizeImageExportSettings } from '@features/gallery/contracts';
import { galleryImageExportNamesOptions, galleryImageExportOptions } from '@features/gallery/queries';
import { downloadBlob } from '@platform/browser/downloadBlob';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useMemo } from 'react';

/** Preview's shortcut for opening the export dialog on the image it shows. */
export const EXPORT_IMAGE_COMMAND_ID = 'viewer.exportImage';

/** One image exports as a file; a selection as archives of its images, its videos left out. */
export type ImageExportTarget =
  | { kind: 'image'; image: GalleryImage }
  | { kind: 'selection'; imageNames: readonly string[]; videoCount: number };

/** Runs an export: rejects for one image, so its dialog can show why; a selection reports through notifications. */
export type ExecuteImageExport = (target: ImageExportTarget, settings: ImageExportSettings) => Promise<void>;

export interface ImageExportRequestOptions {
  /** Where focus goes when the dialog closes; without it, the element focused as it opened, while it stays. */
  returnFocus?: () => HTMLElement | null;
}

export type RequestImageExport = (
  target: ImageExportTarget,
  execute: ExecuteImageExport,
  options?: ImageExportRequestOptions
) => void;

export const getImageExportTargetNames = (target: ImageExportTarget): readonly string[] =>
  target.kind === 'image' ? [target.image.imageName] : target.imageNames;

export const useImageExportPreference = (): ImageExportSettings => {
  const stored = useWorkbenchPreferenceSelector((preferences) => preferences.imageExportSettings);

  return useMemo(() => normalizeImageExportSettings(stored), [stored]);
};

/** Saves the encode a preview of the same settings already holds, else fetches it, under the server's file name. */
export const saveImageExport = async ({
  image,
  owner,
  queryClient,
  settings,
}: {
  image: GalleryImage;
  owner: AccountScope;
  queryClient: QueryClient;
  settings: ImageExportSettings;
}): Promise<void> => {
  const [blob, fileNames] = await Promise.all([
    queryClient.fetchQuery(galleryImageExportOptions(owner, image.imageName, getImageExportEncoding(settings, image))),
    queryClient.fetchQuery(
      galleryImageExportNamesOptions(owner, {
        format: settings.format,
        imageNames: [image.imageName],
        pattern: settings.namePattern,
      })
    ),
  ]);
  const fileName = fileNames[0];

  assertAccountScopeCurrent(owner);
  if (!fileName) {
    throw new Error('The server named no file for the export.');
  }
  await downloadBlob(blob, fileName);
};

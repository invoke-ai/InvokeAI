import type { GalleryImage } from '@features/gallery';
import type { ImageExportSettings } from '@features/gallery/contracts';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { QueryClient, QueryKey } from '@tanstack/react-query';
import type { TFunction } from 'i18next';

import { galleryTransfers } from '@features/gallery';
import {
  getImageExportEncoding,
  IMAGE_EXPORT_FORMAT_LABELS,
  isLossyImageExportFormat,
  normalizeImageExportSettings,
} from '@features/gallery/contracts';
import { galleryImageExportNamesOptions, galleryImageExportOptions } from '@features/gallery/queries';
import { downloadBlob } from '@platform/browser/downloadBlob';
import { formatCount } from '@platform/i18n/languages';
import { useMountEffect } from '@platform/react/useMountEffect';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { getSelectedGalleryImage } from './selectedImage';

export const EXPORT_IMAGE_COMMAND_ID = 'viewer.exportImage';

export type ImageExportTarget =
  | { kind: 'image'; image: GalleryImage }
  | { kind: 'selection'; imageNames: readonly string[]; videoCount: number };

export type ExecuteImageExport = (
  target: ImageExportTarget,
  settings: ImageExportSettings,
  options?: { signal?: AbortSignal }
) => Promise<void>;

export interface ImageExportRequestOptions {
  returnFocus?: () => HTMLElement | null;
}

export type RequestImageExport = (
  target: ImageExportTarget,
  execute: ExecuteImageExport,
  options?: ImageExportRequestOptions
) => void;

export type OpenImageExport = (target: ImageExportTarget, options?: ImageExportRequestOptions) => void;

export const getImageExportTargetNames = (target: ImageExportTarget): readonly string[] =>
  target.kind === 'image' ? [target.image.imageName] : target.imageNames;

export const useImageExportPreference = (): ImageExportSettings => {
  const stored = useWorkbenchPreferenceSelector((preferences) => preferences.imageExportSettings);

  return useMemo(() => normalizeImageExportSettings(stored), [stored]);
};

export const formatImageExportEncoding = (t: TFunction, settings: ImageExportSettings): string =>
  isLossyImageExportFormat(settings.format)
    ? t('widgets.gallery.export.encodingLossy', {
        format: IMAGE_EXPORT_FORMAT_LABELS[settings.format],
        quality: settings.quality,
      })
    : IMAGE_EXPORT_FORMAT_LABELS[settings.format];

export const formatImageExportSummary = (t: TFunction, settings: ImageExportSettings): string =>
  settings.longEdge === null
    ? formatImageExportEncoding(t, settings)
    : t('widgets.gallery.export.summaryWithLimit', {
        encoding: formatImageExportEncoding(t, settings),
        size: formatCount(settings.longEdge),
      });

/** Settled data only: an entry still fetching may be cancelled by the observer that started it. */
const getSettledQueryData = <Data>(queryClient: QueryClient, queryKey: QueryKey): Data | undefined => {
  const query = queryClient.getQueryCache().find<Data>({ exact: true, queryKey });

  return query?.state.status === 'success' && query.state.fetchStatus === 'idle' && !query.isStale()
    ? query.state.data
    : undefined;
};

export const saveImageExport = async ({
  image,
  owner,
  queryClient,
  settings,
  signal,
}: {
  image: GalleryImage;
  owner: AccountScope;
  queryClient: QueryClient;
  settings: ImageExportSettings;
  signal?: AbortSignal;
}): Promise<string> => {
  const requestSignal = signal ? AbortSignal.any([signal, owner.signal]) : owner.signal;
  const encoding = getImageExportEncoding(settings, image);
  const namesRequest = { format: settings.format, imageNames: [image.imageName], pattern: settings.namePattern };
  const [blob, fileNames] = await Promise.all([
    getSettledQueryData<Blob>(queryClient, galleryImageExportOptions(owner, image.imageName, encoding).queryKey) ??
      galleryTransfers.exportImage(image.imageName, encoding, requestSignal),
    getSettledQueryData<string[]>(queryClient, galleryImageExportNamesOptions(owner, namesRequest).queryKey) ??
      galleryTransfers.resolveExportFileNames(namesRequest, requestSignal),
  ]);
  const fileName = fileNames[0];

  assertAccountScopeCurrent(owner);
  requestSignal.throwIfAborted();
  if (!fileName) {
    throw new Error('The server named no file for the export.');
  }
  await downloadBlob(blob, fileName);
  return fileName;
};

/** Registers once: `openImageExport` must be stable, and the caller keys this mount by project and language. */
export const SelectedImageExportCommand = ({
  openImageExport,
  runtime,
}: {
  openImageExport: OpenImageExport;
  runtime: { commands: { register(command: { handler: () => void; id: string; title: string }): () => void } };
}) => {
  const { t } = useTranslation();
  const queries = useWorkbenchQueries();

  useMountEffect(() =>
    runtime.commands.register({
      handler: () => {
        const image = getSelectedGalleryImage(queries.getSnapshot().activeProject);

        if (image) {
          openImageExport({ image, kind: 'image' });
        }
      },
      id: EXPORT_IMAGE_COMMAND_ID,
      title: t('widgets.preview.commands.exportImage'),
    })
  );

  return null;
};

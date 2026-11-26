import type { GalleryItem } from '@features/gallery/contracts';
import type { ResolvedValues } from '@platform/core/resolvedValues';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { QueryClient } from '@tanstack/react-query';
import type {
  WorkbenchGalleryCommands,
  WorkbenchNotificationCommands,
  WorkbenchQueries,
} from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { compareGalleryItems, isGalleryImageItem } from '@features/gallery/contracts';
import { uploadGalleryFiles } from '@features/gallery/picker';
import { resolveGalleryUploadBoardId } from '@features/gallery/queries';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';

import type { PreviewDropZone } from './PreviewDropSurface';

/** Uploads like the Gallery, then shows or compares them in the drop's project; account changes drop them. */
export const importPreviewFileDrop = async ({
  files,
  gallery,
  galleryValues,
  notifications,
  owner,
  projectId,
  queries,
  queryClient,
  t,
  zone,
}: {
  files: File[];
  gallery: Pick<WorkbenchGalleryCommands, 'selectItemInItsBoard' | 'setCompareItem'>;
  /** The project's gallery values when the files were dropped, which name its auto-add board. */
  galleryValues: ResolvedValues;
  notifications: WorkbenchNotificationCommands;
  /** The account the files were dropped in. */
  owner: AccountScope;
  projectId: string;
  queries: Pick<WorkbenchQueries, 'getProject'>;
  queryClient: QueryClient;
  t: TFunction;
  zone: PreviewDropZone;
}): Promise<void> => {
  const boardId = await resolveGalleryUploadBoardId(galleryValues, queryClient);
  if (!isAccountScopeCurrent(owner)) {
    return;
  }
  const uploaded = await uploadGalleryFiles(files, {
    boards: [],
    notifications,
    queryClient,
    selectedBoardId: boardId,
    t,
  });
  // Nothing stored (the upload said why, or its account is gone), or the project was closed meanwhile.
  if (uploaded.length === 0 || !queries.getProject(projectId)) {
    return;
  }
  const comparison = zone === 'compare' ? uploaded.find(isGalleryImageItem) : undefined;
  if (comparison) {
    gallery.setCompareItem(comparison, projectId);
    return;
  }
  // The newest upload, as the Gallery selects after an upload of its own.
  const newest = uploaded.reduce<GalleryItem>(
    (current, item) => (compareGalleryItems(item, current, { orderDir: 'DESC' }) < 0 ? item : current),
    uploaded[0]!
  );
  gallery.selectItemInItsBoard(newest, projectId);
};

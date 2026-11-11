import type { GalleryItem } from '@features/gallery/contracts';
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

/**
 * Uploads media files dropped on Preview as the Gallery uploads them (the project's auto-add board, the upload's own
 * notices, unsupported files refused there) and then shows them: the newest upload is selected within its own board,
 * as a pick from Preview's mini gallery is, or on the compare zone the first image becomes the comparison. Both go to
 * the project the files were dropped on, so a switch made meanwhile leaves the open project's Preview alone; an
 * account change on the way leaves nothing behind.
 */
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
  galleryValues: Record<string, unknown>;
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

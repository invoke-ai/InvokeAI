import type { ResolvedValues } from '@platform/core/resolvedValues';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { QueryClient } from '@tanstack/react-query';
import type {
  WorkbenchCanvasCommands,
  WorkbenchNotificationCommands,
  WorkbenchQueries,
} from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { galleryImageItemToGalleryImage, isGalleryImageItem } from '@features/gallery/contracts';
import { uploadGalleryFiles } from '@features/gallery/picker';
import { resolveGalleryUploadBoardId } from '@features/gallery/queries';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getCanvasEngine, getCanvasImportNotice } from '@workbench/canvas-operations/api';

import type { CanvasImageDropData } from './canvasImageDnd';

import { importCanvasDropImages } from './executeCanvasImageDropImport';

/** Uploads dropped files like pasted media, then lands them in the drop's project, never the current one. */
export const importCanvasFileDrop = async ({
  canvas,
  destination,
  files,
  galleryValues,
  notifications,
  owner,
  projectId,
  queries,
  queryClient,
  t,
}: {
  canvas: WorkbenchCanvasCommands;
  destination: CanvasImageDropData['destination'];
  files: File[];
  /** The project's gallery values when the files were dropped, which name its auto-add board. */
  galleryValues: ResolvedValues;
  notifications: WorkbenchNotificationCommands;
  /** The account the files were dropped in. */
  owner: AccountScope;
  projectId: string;
  queries: Pick<WorkbenchQueries, 'getProject' | 'isActiveProject'>;
  queryClient: QueryClient;
  t: TFunction;
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
  // Nothing stored: the upload said why, or the account it belonged to is gone.
  const images = uploaded.filter(isGalleryImageItem).map(galleryImageItemToGalleryImage);
  if (images.length === 0) {
    return;
  }
  const project = queries.getProject(projectId);
  const notice = project
    ? await importCanvasDropImages({
        canvas,
        destination,
        engine: getCanvasEngine(projectId) ?? null,
        images,
        project,
        queries,
      })
    : getCanvasImportNotice({ status: 'stale-project' });
  notifications.add({ kind: notice.kind, title: t(notice.titleKey, notice.options ?? {}) });
};

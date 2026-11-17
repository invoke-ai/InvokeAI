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

/**
 * Lands image files dropped on a canvas zone as that zone lands gallery images. They are uploaded first, as pasted
 * media is: to the project's auto-add board, reporting the upload's outcome and failures. The images then go to the
 * project the drop was made on, read again once they are stored so edits made meanwhile are kept, through the canvas
 * engine live for it then; a project switched away from, or one whose Canvas was closed meanwhile, takes them through
 * its ingestion, never the canvas now showing. An account change on the way leaves nothing behind.
 */
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
  galleryValues: Record<string, unknown>;
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

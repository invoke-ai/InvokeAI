import { useMemo, type Dispatch } from 'react';

import { saveBlobToDisk } from '@workbench/components/useImageActions';
import {
  createGalleryBoard,
  deleteGalleryBoard,
  downloadGalleryArchive,
  updateGalleryBoard,
  uploadGalleryImage,
  type GalleryBoard,
} from '@workbench/gallery/api';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { GalleryActions } from './GalleryWidgetContext';

const ACCEPTED_UPLOAD_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export const useGalleryActions = ({
  boards,
  dispatch,
  loadMore,
  projectBoardId,
  projectName,
  selectedBoardId,
}: {
  boards: GalleryBoard[];
  dispatch: Dispatch<WorkbenchAction>;
  loadMore: () => void;
  projectBoardId: string | null;
  projectName: string;
  selectedBoardId: string;
}): GalleryActions => {
  return useMemo<GalleryActions>(() => {
    const recordError = (error: unknown) => dispatch({ message: toErrorMessage(error), type: 'recordError' });
    const recordSuccess = (title: string, message?: string) =>
      dispatch({ kind: 'success', message, title, type: 'recordNotice' });
    const refresh = () => dispatch({ type: 'touchGalleryRefresh' });
    const getBoardName = (boardId: string) => boards.find((board) => board.id === boardId)?.name ?? 'Uncategorized';

    return {
      archiveBoard: async (boardId, archived) => {
        try {
          await updateGalleryBoard(boardId, { archived });
          recordSuccess(`${archived ? 'Archived' : 'Unarchived'} board "${getBoardName(boardId)}"`);

          if (archived && boardId === selectedBoardId) {
            dispatch({ boardId: 'none', type: 'selectGalleryBoard' });
          }

          refresh();
        } catch (error: unknown) {
          recordError(error);
        }
      },
      createBoard: async (boardName) => {
        try {
          const board = await createGalleryBoard(boardName);

          dispatch({ boardId: board.id, type: 'selectGalleryBoard' });
          recordSuccess(`Created board "${board.name}"`);
          refresh();
        } catch (error: unknown) {
          recordError(error);
        }
      },
      deleteBoard: async (boardId, includeImages) => {
        try {
          const boardName = getBoardName(boardId);

          await deleteGalleryBoard(boardId, includeImages);
          recordSuccess(
            `Deleted board "${boardName}"`,
            includeImages ? 'Its images were permanently deleted.' : 'Its images were moved to Uncategorized.'
          );

          if (boardId === selectedBoardId) {
            dispatch({ boardId: 'none', type: 'selectGalleryBoard' });
          }

          refresh();
        } catch (error: unknown) {
          recordError(error);
        }
      },
      downloadBoard: async (boardId) => {
        try {
          dispatch({
            kind: 'info',
            message: `Preparing an archive of "${getBoardName(boardId)}".`,
            title: 'Preparing download',
            type: 'recordNotice',
          });

          const { blob, fileName } = await downloadGalleryArchive({ boardId });

          saveBlobToDisk(blob, fileName);
          recordSuccess('Download ready');
        } catch (error: unknown) {
          recordError(error);
        }
      },
      loadMore,
      refresh,
      renameBoard: async (boardId, boardName) => {
        try {
          await updateGalleryBoard(boardId, { name: boardName });
          recordSuccess(`Renamed board to "${boardName}"`);
          refresh();
        } catch (error: unknown) {
          recordError(error);
        }
      },
      selectBoard: (boardId) => dispatch({ boardId, type: 'selectGalleryBoard' }),
      selectImage: (image) => dispatch({ image, type: 'selectGalleryImage' }),
      selectImageRange: (imageNames, primaryImage) =>
        dispatch({ imageNames, primaryImage, type: 'setGalleryMultiSelection' }),
      selectProjectBoard: async () => {
        if (projectBoardId && boards.some((board) => board.id === projectBoardId)) {
          dispatch({ boardId: projectBoardId, type: 'selectGalleryBoard' });
          return;
        }

        if (boards.length === 0) {
          return;
        }

        try {
          const board = await createGalleryBoard(projectName);

          dispatch({ boardId: board.id, type: 'setGalleryProjectBoardId' });
          dispatch({ boardId: board.id, type: 'selectGalleryBoard' });
          recordSuccess(`Created project board "${board.name}"`);
          refresh();
        } catch (error: unknown) {
          recordError(error);
        }
      },
      setSearchTerm: (searchTerm) => dispatch({ searchTerm, type: 'setGallerySearchTerm' }),
      setView: (galleryView) => dispatch({ galleryView, type: 'setGalleryView' }),
      toggleImageInSelection: (image) => dispatch({ image, type: 'toggleGalleryImageInSelection' }),
      updateSettings: (settings) => dispatch({ settings, type: 'updateGallerySettings' }),
      uploadFiles: async (files) => {
        const accepted = files.filter((file) => ACCEPTED_UPLOAD_TYPES.has(file.type));

        if (accepted.length === 0) {
          dispatch({ message: 'No supported image files to upload (PNG, JPEG, or WebP).', type: 'recordError' });
          return;
        }

        try {
          await Promise.all(accepted.map((file) => uploadGalleryImage(file, selectedBoardId)));
          recordSuccess(
            `Uploaded ${accepted.length} ${accepted.length === 1 ? 'image' : 'images'}`,
            `Added to ${getBoardName(selectedBoardId)} as assets.`
          );
          refresh();
        } catch (error: unknown) {
          recordError(error);
        }
      },
    };
  }, [boards, dispatch, loadMore, projectBoardId, projectName, selectedBoardId]);
};

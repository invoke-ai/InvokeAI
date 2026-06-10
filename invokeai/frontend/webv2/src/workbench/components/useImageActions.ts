import { useMemo, type Dispatch } from 'react';

import {
  addImagesToGalleryBoard,
  deleteGalleryImages,
  downloadGalleryArchive,
  removeImagesFromGalleryBoard,
  starGalleryImages,
  unstarGalleryImages,
  type GalleryBoard,
  type GalleryImage,
} from '../gallery/api';
import type { WorkbenchAction } from '../workbenchState';

/**
 * Image operations shared by every surface that shows backend images (gallery
 * grid, preview, image context menus). Each mutation notifies the gallery via
 * `touchGalleryRefresh` so any mounted gallery widget refetches.
 */
export interface ImageActions {
  copyImage: (image: GalleryImage) => Promise<void>;
  deleteImages: (imageNames: string[]) => Promise<void>;
  downloadImage: (image: GalleryImage) => Promise<void>;
  downloadImages: (imageNames: string[]) => Promise<void>;
  moveImagesToBoard: (imageNames: string[], boardId: string) => Promise<void>;
  openImageInPreview: (image: GalleryImage) => void;
  selectForCompare: (image: GalleryImage) => void;
  setImagesStarred: (imageNames: string[], starred: boolean) => Promise<void>;
}

export const saveBlobToDisk = (blob: Blob, fileName: string): void => {
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement('a');

  anchor.href = objectUrl;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(objectUrl);
};

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

const toPngBlob = async (blob: Blob): Promise<Blob> => {
  if (blob.type === 'image/png') {
    return blob;
  }

  const bitmap = await createImageBitmap(blob);
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext('2d')?.drawImage(bitmap, 0, 0);

  return new Promise((resolve, reject) => {
    canvas.toBlob((pngBlob) => (pngBlob ? resolve(pngBlob) : reject(new Error('Failed to encode PNG.'))), 'image/png');
  });
};

export const useImageActions = ({
  boards,
  dispatch,
  onImagesDeleted,
  onStarredChange,
}: {
  boards: GalleryBoard[];
  dispatch: Dispatch<WorkbenchAction>;
  /** Called after a successful deletion so the host can select a neighboring image. */
  onImagesDeleted?: (imageNames: string[]) => void;
  /** Optional optimistic hook, called before the request and re-called inverted on failure. */
  onStarredChange?: (imageNames: string[], starred: boolean) => void;
}): ImageActions => {
  return useMemo<ImageActions>(() => {
    const recordError = (error: unknown) => dispatch({ message: toErrorMessage(error), type: 'recordError' });
    const recordSuccess = (title: string, message?: string) =>
      dispatch({ kind: 'success', message, title, type: 'recordNotice' });
    const refreshGallery = () => dispatch({ type: 'touchGalleryRefresh' });
    const getBoardName = (boardId: string) => boards.find((board) => board.id === boardId)?.name ?? 'Uncategorized';

    return {
      copyImage: async (image) => {
        try {
          const response = await fetch(image.imageUrl);
          const blob = await toPngBlob(await response.blob());

          await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
          recordSuccess('Copied image to clipboard');
        } catch (error: unknown) {
          recordError(error);
        }
      },
      deleteImages: async (imageNames) => {
        try {
          await deleteGalleryImages(imageNames);
          dispatch({ imageNames, type: 'removeGalleryImages' });
          onImagesDeleted?.(imageNames);
          recordSuccess(imageNames.length === 1 ? 'Deleted image' : `Deleted ${imageNames.length} images`);
          refreshGallery();
        } catch (error: unknown) {
          recordError(error);
        }
      },
      downloadImage: async (image) => {
        try {
          const response = await fetch(image.imageUrl);
          const objectUrl = URL.createObjectURL(await response.blob());
          const anchor = document.createElement('a');

          anchor.href = objectUrl;
          anchor.download = image.imageName;
          anchor.click();
          URL.revokeObjectURL(objectUrl);
        } catch (error: unknown) {
          recordError(error);
        }
      },
      downloadImages: async (imageNames) => {
        try {
          dispatch({
            kind: 'info',
            message: `Preparing an archive of ${imageNames.length} images.`,
            title: 'Preparing download',
            type: 'recordNotice',
          });

          const { blob, fileName } = await downloadGalleryArchive({ imageNames });

          saveBlobToDisk(blob, fileName);
          recordSuccess('Download ready');
        } catch (error: unknown) {
          recordError(error);
        }
      },
      moveImagesToBoard: async (imageNames, boardId) => {
        try {
          if (boardId === 'none') {
            await removeImagesFromGalleryBoard(imageNames);
          } else {
            await addImagesToGalleryBoard(boardId, imageNames);
          }

          recordSuccess(
            imageNames.length === 1
              ? `Moved image to ${getBoardName(boardId)}`
              : `Moved ${imageNames.length} images to ${getBoardName(boardId)}`
          );
          refreshGallery();
        } catch (error: unknown) {
          recordError(error);
        }
      },
      openImageInPreview: (image) => {
        dispatch({ image, type: 'selectGalleryImage' });
        dispatch({ centerViewId: 'preview', type: 'setCenterView' });
      },
      selectForCompare: (image) => {
        dispatch({ image, type: 'setGalleryCompareImage' });
      },
      setImagesStarred: async (imageNames, starred) => {
        onStarredChange?.(imageNames, starred);

        try {
          await (starred ? starGalleryImages : unstarGalleryImages)(imageNames);
          refreshGallery();
        } catch (error: unknown) {
          onStarredChange?.(imageNames, !starred);
          recordError(error);
        }
      },
    };
  }, [boards, dispatch, onImagesDeleted, onStarredChange]);
};

import type { VaeModelConfig } from '@workbench/generation/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { importGalleryImagesToCanvas, type GalleryCanvasImportDestination } from '@workbench/canvas-operations/api';
import { getCanvasImportNotice } from '@workbench/canvas-operations/canvasImportNotice';
import { getEngine } from '@workbench/canvas-operations/engineRegistry';
import {
  addImagesToGalleryBoard,
  deleteGalleryImages,
  downloadGalleryArchive,
  getGalleryImageMetadata,
  removeImagesFromGalleryBoard,
  starGalleryImages,
  unstarGalleryImages,
  type GalleryBoard,
  type GalleryImage,
} from '@workbench/gallery/api';
import {
  createReferenceImageId,
  getDefaultReferenceImageConfig,
  getMaxReferenceImages,
  isReferenceImageSupported,
  isSupportedGenerateModel,
} from '@workbench/generation/baseGenerationPolicies';
import { isVaeModelConfig } from '@workbench/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@workbench/models/modelsStore';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchStore } from '@workbench/WorkbenchContext';
import { useEffect, useMemo, type Dispatch } from 'react';
import { useTranslation } from 'react-i18next';

import { recordCanvasImportError } from './canvasImportError';
import { executeImageRecall, getCurrentGenerateValues } from './executeImageRecall';
import {
  EMPTY_IMAGE_RECALL_CAPABILITIES,
  getImageRecallCapabilities,
  type ImageRecallCapabilities,
  type ImageRecallKind,
} from './imageRecall';

/**
 * Image operations shared by every surface that shows backend images (gallery
 * grid, preview, image context menus). Each mutation notifies the gallery via
 * a refresh action so any mounted gallery widget refetches affected backend data.
 */
export interface ImageActions {
  /** Whether the generate widget's current model can accept another reference image. */
  canUseAsReferenceImage: boolean;
  copyImage: (image: GalleryImage) => Promise<void>;
  deleteImages: (imageNames: string[]) => Promise<void>;
  downloadImage: (image: GalleryImage) => Promise<void>;
  downloadImages: (imageNames: string[]) => Promise<void>;
  getImageRecallCapabilities: (image: GalleryImage) => Promise<ImageRecallCapabilities>;
  moveImagesToBoard: (imageNames: string[], boardId: string) => Promise<void>;
  openImageInPreview: (image: GalleryImage) => void;
  recallImageData: (image: GalleryImage, kind: ImageRecallKind) => Promise<void>;
  selectForCompare: (image: GalleryImage) => void;
  sendToCanvas: (images: readonly GalleryImage[], destination: GalleryCanvasImportDestination) => Promise<void>;
  setImagesStarred: (imageNames: string[], starred: boolean) => Promise<void>;
  useAsReferenceImage: (image: GalleryImage) => void;
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
  generateValues,
  onImagesDeleted,
  onStarredChange,
  projectId,
}: {
  boards: GalleryBoard[];
  dispatch: Dispatch<WorkbenchAction>;
  generateValues: Record<string, unknown>;
  projectId?: string;
  /** Called after a successful deletion so the host can select a neighboring image. */
  onImagesDeleted?: (imageNames: string[]) => void;
  /** Optional optimistic hook, called before the request and re-called inverted on failure. */
  onStarredChange?: (imageNames: string[], starred: boolean) => void;
}): ImageActions => {
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const store = useWorkbenchStore();
  const { t } = useTranslation();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const supportedModels = useMemo(() => models.filter(isSupportedGenerateModel), [models]);
  const vaeModels = useMemo(() => models.filter(isVaeModelConfig).map((model) => model as VaeModelConfig), [models]);
  const currentGenerateValues = useMemo(() => {
    return getCurrentGenerateValues({ generateValues, supportedModels });
  }, [generateValues, supportedModels]);

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  return useMemo<ImageActions>(() => {
    const recordError = (error: unknown) =>
      dispatch({
        area: 'image-actions',
        message: toErrorMessage(error),
        namespace: 'gallery',
        projectId,
        type: 'recordError',
      });
    const recordSuccess = (title: string, message?: string) =>
      dispatch({ kind: 'success', message, title, type: 'recordNotice' });
    const refreshGallery = () => dispatch({ projectId, type: 'touchGalleryRefresh' });
    const refreshGalleryImages = () => dispatch({ projectId, type: 'touchGalleryImagesRefresh' });
    const getBoardName = (boardId: string) => boards.find((board) => board.id === boardId)?.name ?? 'Uncategorized';
    const getLatestGenerateValues = () => {
      const snapshot = store.getSnapshot();
      const project = projectId
        ? snapshot.state.projects.find((candidate) => candidate.id === projectId)
        : snapshot.activeProject;

      return project ? getProjectWidgetValues(project, 'generate') : {};
    };

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
          dispatch({ imageNames, projectId, type: 'removeGalleryImages' });
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
      getImageRecallCapabilities: async (image) => {
        if (!currentGenerateValues) {
          return EMPTY_IMAGE_RECALL_CAPABILITIES;
        }

        try {
          const metadata = await getGalleryImageMetadata(image.imageName);

          return getImageRecallCapabilities({
            currentValues: currentGenerateValues,
            image,
            metadata,
            models,
            supportedModels,
            vaeModels,
          });
        } catch {
          return {
            ...EMPTY_IMAGE_RECALL_CAPABILITIES,
            dimensions:
              Number.isFinite(image.width) && image.width >= 64 && Number.isFinite(image.height) && image.height >= 64,
          };
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
        dispatch({ image, projectId, type: 'selectGalleryImage' });
        openWorkbenchWidget('preview', { preferredRegions: ['center'], requireCenterView: true });
      },
      recallImageData: async (image, kind) => {
        const didRecall = await executeImageRecall({
          dispatch,
          generateValues,
          getGenerateValues: getLatestGenerateValues,
          image,
          kind,
          models,
          projectId,
        });

        if (didRecall && (!projectId || store.getSnapshot().activeProject.id === projectId)) {
          openWorkbenchWidget('generate', { preferredRegions: ['left'] });
        }
      },
      selectForCompare: (image) => {
        dispatch({ image, projectId, type: 'setGalleryCompareImage' });
      },
      sendToCanvas: async (images, destination) => {
        try {
          const state = store.getState();
          const targetProjectId = projectId ?? state.activeProjectId;
          const project = state.projects.find((candidate) => candidate.id === targetProjectId);

          if (!project) {
            const notice = getCanvasImportNotice({ status: 'stale-project' });
            dispatch({ kind: notice.kind, title: t(notice.titleKey, notice.options ?? {}), type: 'recordNotice' });
            return;
          }

          const result = await importGalleryImagesToCanvas({
            destination,
            dispatch,
            engine: getEngine(project.id) ?? null,
            getState: store.getState,
            images,
            project,
          });
          const notice = getCanvasImportNotice(result);
          dispatch({ kind: notice.kind, title: t(notice.titleKey, notice.options ?? {}), type: 'recordNotice' });

          if (result.status === 'imported' && store.getState().activeProjectId === project.id) {
            openWorkbenchWidget('canvas', { preferredRegions: ['center'], requireCenterView: true });
          }
        } catch (error: unknown) {
          recordCanvasImportError({
            dispatch,
            error,
            localizedMessage: t('widgets.canvas.import.failed'),
            projectId,
          });
        }
      },
      setImagesStarred: async (imageNames, starred) => {
        onStarredChange?.(imageNames, starred);

        try {
          await (starred ? starGalleryImages : unstarGalleryImages)(imageNames);
          refreshGalleryImages();
        } catch (error: unknown) {
          onStarredChange?.(imageNames, !starred);
          recordError(error);
        }
      },
      canUseAsReferenceImage: Boolean(
        currentGenerateValues &&
        currentGenerateValues.referenceImages.length < getMaxReferenceImages(currentGenerateValues.model)
      ),
      useAsReferenceImage: (image) => {
        const latestGenerateValues = getLatestGenerateValues();
        const currentValues = getCurrentGenerateValues({ generateValues: latestGenerateValues, supportedModels });

        if (
          !currentValues ||
          !isReferenceImageSupported(currentValues.model) ||
          currentValues.referenceImages.length >= getMaxReferenceImages(currentValues.model)
        ) {
          return;
        }

        const referenceImage = {
          config: getDefaultReferenceImageConfig(currentValues.model, models, image),
          id: createReferenceImageId(),
          isEnabled: true,
        };

        dispatch({
          projectId,
          type: 'patchGenerateSettings',
          values: { referenceImages: [...currentValues.referenceImages, referenceImage] },
        });
        openWorkbenchWidget('generate', { preferredRegions: ['left'] });
      },
    };
  }, [
    boards,
    currentGenerateValues,
    dispatch,
    generateValues,
    models,
    onImagesDeleted,
    onStarredChange,
    openWorkbenchWidget,
    projectId,
    store,
    supportedModels,
    t,
    vaeModels,
  ]);
};

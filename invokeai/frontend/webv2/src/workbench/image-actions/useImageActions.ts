import type { VaeModelConfig } from '@features/generation/contracts';

import {
  galleryImages,
  galleryOrganization,
  galleryTransfers,
  type GalleryBoard,
  type GalleryImage,
} from '@features/gallery';
import {
  createReferenceImageId,
  generatedImageToReferenceImage,
  getDefaultReferenceImageConfig,
  getMaxReferenceImages,
  isVaeModelConfig,
  isReferenceImageSupported,
  isSupportedGenerateModel,
} from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  getCanvasImportNotice,
  getCanvasEngine,
  importGalleryImagesToCanvas,
  type GalleryCanvasImportDestination,
} from '@workbench/canvas-operations/api';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';
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
  generateValues,
  onImagesDeleted,
  onStarredChange,
  projectId,
}: {
  boards: GalleryBoard[];
  generateValues: Record<string, unknown>;
  projectId?: string;
  /** Called after a successful deletion so the host can select a neighboring image. */
  onImagesDeleted?: (imageNames: string[]) => void;
  /** Optional optimistic hook, called before the request and re-called inverted on failure. */
  onStarredChange?: (imageNames: string[], starred: boolean) => void;
}): ImageActions => {
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const commands = useWorkbenchCommands();
  const { gallery, generation, notifications } = commands;
  const queries = useWorkbenchQueries();
  const { t } = useTranslation();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const supportedModels = useMemo(() => models.filter(isSupportedGenerateModel), [models]);
  const vaeModels = useMemo(() => models.filter(isVaeModelConfig).map((model) => model as VaeModelConfig), [models]);
  const currentGenerateValues = useMemo(() => {
    return getCurrentGenerateValues({ generateValues, supportedModels });
  }, [generateValues, supportedModels]);

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  return useMemo<ImageActions>(() => {
    const recordError = (error: unknown) =>
      notifications.reportError({
        area: 'image-actions',
        message: toErrorMessage(error),
        namespace: 'gallery',
        projectId,
      });
    const recordSuccess = (title: string, message?: string) => notifications.add({ kind: 'success', message, title });
    const refreshGallery = () => gallery.touch(projectId);
    const refreshGalleryImages = () => gallery.touchImages(projectId);
    const getBoardName = (boardId: string) => boards.find((board) => board.id === boardId)?.name ?? 'Uncategorized';
    const getLatestGenerateValues = () => {
      const snapshot = queries.getSnapshot();
      const project = projectId
        ? snapshot.projects.find((candidate) => candidate.id === projectId)
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
          await galleryOrganization.deleteImages(imageNames);
          gallery.removeImages(imageNames, projectId);
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
          notifications.add({
            kind: 'info',
            message: `Preparing an archive of ${imageNames.length} images.`,
            title: 'Preparing download',
          });

          const { blob, fileName } = await galleryTransfers.downloadArchive({ imageNames });

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
          const metadata = await galleryImages.metadata(image.imageName);

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
            await galleryOrganization.removeFromBoard(imageNames);
          } else {
            await galleryOrganization.addToBoard(boardId, imageNames);
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
        gallery.selectImage(image, projectId);
        openWorkbenchWidget('preview', { preferredRegions: ['center'], requireCenterView: true });
      },
      recallImageData: async (image, kind) => {
        const didRecall = await executeImageRecall({
          commands,
          generateValues,
          getGenerateValues: getLatestGenerateValues,
          image,
          kind,
          models,
          projectId,
        });

        if (didRecall && (!projectId || queries.isActiveProject(projectId))) {
          openWorkbenchWidget('generate', { preferredRegions: ['left'] });
        }
      },
      selectForCompare: (image) => {
        gallery.setCompareImage(image, projectId);
      },
      sendToCanvas: async (images, destination) => {
        try {
          const targetProjectId = projectId ?? queries.getSnapshot().activeProject.id;
          const project = queries.getProject(targetProjectId);

          if (!project) {
            const notice = getCanvasImportNotice({ status: 'stale-project' });
            notifications.add({ kind: notice.kind, title: t(notice.titleKey, notice.options ?? {}) });
            return;
          }

          const result = await importGalleryImagesToCanvas({
            applyCanvasMutation: commands.canvas.apply,
            destination,
            engine: getCanvasEngine(project.id) ?? null,
            getProject: queries.getProject,
            images,
            isActiveProject: queries.isActiveProject,
            project,
          });
          const notice = getCanvasImportNotice(result);
          notifications.add({ kind: notice.kind, title: t(notice.titleKey, notice.options ?? {}) });

          if (result.status === 'imported' && queries.isActiveProject(project.id)) {
            openWorkbenchWidget('canvas', { preferredRegions: ['center'], requireCenterView: true });
          }
        } catch (error: unknown) {
          recordCanvasImportError({
            error,
            localizedMessage: t('widgets.canvas.import.failed'),
            notifications,
            projectId,
          });
        }
      },
      setImagesStarred: async (imageNames, starred) => {
        onStarredChange?.(imageNames, starred);

        try {
          await galleryOrganization.setStarred(imageNames, starred);
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
          config: getDefaultReferenceImageConfig(currentValues.model, models, generatedImageToReferenceImage(image)),
          id: createReferenceImageId(),
          isEnabled: true,
        };

        generation.patchSettings({ referenceImages: [...currentValues.referenceImages, referenceImage] }, projectId);
        openWorkbenchWidget('generate', { preferredRegions: ['left'] });
      },
    };
  }, [
    boards,
    currentGenerateValues,
    commands,
    gallery,
    generateValues,
    generation,
    models,
    notifications,
    onImagesDeleted,
    onStarredChange,
    openWorkbenchWidget,
    projectId,
    queries,
    supportedModels,
    t,
    vaeModels,
  ]);
};

import type { VaeModelConfig } from '@features/generation/contracts';

import {
  galleryImages,
  galleryOrganization,
  galleryTransfers,
  type GalleryBoard,
  type GalleryImage,
} from '@features/gallery';
import { invalidateGallery, invalidateGalleryImages, patchGalleryImageCaches } from '@features/gallery/queries';
import { setPendingPromptTemplateDraft } from '@features/generation/react';
import { getMaxReferenceImages, isVaeModelConfig, isSupportedGenerateModel } from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { useQueryClient } from '@tanstack/react-query';
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

import { appendReferenceImage } from './appendReferenceImage';
import { recordCanvasImportError } from './canvasImportError';
import { executeImageRecall, getCurrentGenerateValues } from './executeImageRecall';
import {
  getMetadataPrompts,
  EMPTY_IMAGE_RECALL_CAPABILITIES,
  getImageRecallCapabilities,
  type ImageRecallCapabilities,
  type ImageRecallKind,
} from './imageRecall';

/**
 * Image operations shared by every surface that shows backend images (gallery
 * grid, preview, image context menus). Mutations patch the shared Gallery cache
 * when possible and explicitly invalidate the affected server state.
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
  /** Opens the generate widget's template editor prefilled from this image's prompts. */
  savePromptAsTemplate: (image: GalleryImage) => Promise<void>;
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
  projectId,
}: {
  boards: GalleryBoard[];
  generateValues: Record<string, unknown>;
  projectId?: string;
  /** Called after a successful deletion so the host can select a neighboring image. */
  onImagesDeleted?: (imageNames: string[]) => void;
}): ImageActions => {
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const commands = useWorkbenchCommands();
  const { gallery, generation, notifications } = commands;
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();
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
        const owner = captureAccountScope();

        try {
          const response = await fetch(image.imageUrl, { signal: owner.signal });
          const sourceBlob = await response.blob();

          assertAccountScopeCurrent(owner);
          const blob = await toPngBlob(sourceBlob);

          assertAccountScopeCurrent(owner);
          await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
          assertAccountScopeCurrent(owner);
          recordSuccess('Copied image to clipboard');
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
      deleteImages: async (imageNames) => {
        const owner = captureAccountScope();

        try {
          await galleryOrganization.deleteImages(imageNames, owner.signal);

          assertAccountScopeCurrent(owner);
          patchGalleryImageCaches(queryClient, { imageNames, kind: 'delete' });
          gallery.removeImages(imageNames);
          onImagesDeleted?.(imageNames);
          recordSuccess(imageNames.length === 1 ? 'Deleted image' : `Deleted ${imageNames.length} images`);
          void invalidateGallery(queryClient);
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
      downloadImage: async (image) => {
        const owner = captureAccountScope();

        try {
          const response = await fetch(image.imageUrl, { signal: owner.signal });
          const blob = await response.blob();

          assertAccountScopeCurrent(owner);
          const objectUrl = URL.createObjectURL(blob);
          const anchor = document.createElement('a');

          anchor.href = objectUrl;
          anchor.download = image.imageName;
          anchor.click();
          URL.revokeObjectURL(objectUrl);
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
      downloadImages: async (imageNames) => {
        const owner = captureAccountScope();

        try {
          notifications.add({
            kind: 'info',
            message: `Preparing an archive of ${imageNames.length} images.`,
            title: 'Preparing download',
          });

          const { blob, fileName } = await galleryTransfers.downloadArchive({ imageNames, signal: owner.signal });

          assertAccountScopeCurrent(owner);
          saveBlobToDisk(blob, fileName);
          recordSuccess('Download ready');
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
      getImageRecallCapabilities: async (image) => {
        const owner = captureAccountScope();

        if (!currentGenerateValues) {
          return EMPTY_IMAGE_RECALL_CAPABILITIES;
        }

        try {
          const metadata = await galleryImages.metadata(image.imageName, owner.signal);

          assertAccountScopeCurrent(owner);
          return getImageRecallCapabilities({
            currentValues: currentGenerateValues,
            image,
            metadata,
            models,
            supportedModels,
            vaeModels,
          });
        } catch {
          if (!isAccountScopeCurrent(owner)) {
            return EMPTY_IMAGE_RECALL_CAPABILITIES;
          }

          return {
            ...EMPTY_IMAGE_RECALL_CAPABILITIES,
            dimensions:
              Number.isFinite(image.width) && image.width >= 64 && Number.isFinite(image.height) && image.height >= 64,
          };
        }
      },
      savePromptAsTemplate: async (image) => {
        const owner = captureAccountScope();

        try {
          const metadata = await galleryImages.metadata(image.imageName, owner.signal);

          assertAccountScopeCurrent(owner);

          const { negativePrompt, positivePrompt } = getMetadataPrompts(metadata);

          if (!positivePrompt && !negativePrompt) {
            notifications.add({ kind: 'info', title: 'This image has no prompt to save' });
            return;
          }

          // The widget hosts the editor, so it has to be on screen for the
          // handoff to be visible.
          openWorkbenchWidget('generate', { preferredRegions: ['left'] });
          setPendingPromptTemplateDraft({ negativePrompt, positivePrompt });
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
      moveImagesToBoard: async (imageNames, boardId) => {
        const owner = captureAccountScope();

        try {
          if (boardId === 'none') {
            await galleryOrganization.removeFromBoard(imageNames, owner.signal);
          } else {
            await galleryOrganization.addToBoard(boardId, imageNames, owner.signal);
          }

          assertAccountScopeCurrent(owner);
          patchGalleryImageCaches(queryClient, { boardId, imageNames, kind: 'move' });
          gallery.patchImages(imageNames, { boardId });
          recordSuccess(
            imageNames.length === 1
              ? `Moved image to ${getBoardName(boardId)}`
              : `Moved ${imageNames.length} images to ${getBoardName(boardId)}`
          );
          void invalidateGallery(queryClient);
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
      openImageInPreview: (image) => {
        gallery.selectImage(image, projectId);
        openWorkbenchWidget('preview', { preferredRegions: ['center'], requireCenterView: true });
      },
      recallImageData: async (image, kind) => {
        const owner = captureAccountScope();
        const didRecall = await executeImageRecall({
          commands,
          generateValues,
          getGenerateValues: getLatestGenerateValues,
          image,
          kind,
          models,
          projectId,
        });

        if (isAccountScopeCurrent(owner) && didRecall && (!projectId || queries.isActiveProject(projectId))) {
          openWorkbenchWidget('generate', { preferredRegions: ['left'] });
        }
      },
      selectForCompare: (image) => {
        gallery.setCompareImage(image, projectId);
      },
      sendToCanvas: async (images, destination) => {
        const owner = captureAccountScope();

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

          assertAccountScopeCurrent(owner);
          const notice = getCanvasImportNotice(result);
          notifications.add({ kind: notice.kind, title: t(notice.titleKey, notice.options ?? {}) });

          if (result.status === 'imported' && queries.isActiveProject(project.id)) {
            openWorkbenchWidget('canvas', { preferredRegions: ['center'], requireCenterView: true });
          }
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordCanvasImportError({
            error,
            localizedMessage: t('widgets.canvas.import.failed'),
            notifications,
            projectId,
          });
        }
      },
      setImagesStarred: async (imageNames, starred) => {
        const owner = captureAccountScope();
        const rollback = patchGalleryImageCaches(queryClient, { imageNames, kind: 'star', starred });

        try {
          await galleryOrganization.setStarred(imageNames, starred, owner.signal);

          assertAccountScopeCurrent(owner);
          gallery.patchImages(imageNames, { starred });
          void invalidateGalleryImages(queryClient);
        } catch (error: unknown) {
          rollback();

          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          void invalidateGalleryImages(queryClient);
          recordError(error);
        }
      },
      canUseAsReferenceImage: Boolean(
        currentGenerateValues &&
        currentGenerateValues.referenceImages.length < getMaxReferenceImages(currentGenerateValues.model)
      ),
      useAsReferenceImage: (image) => {
        const result = appendReferenceImage({ generateValues: getLatestGenerateValues(), image, models });

        if (result.status !== 'appended') {
          return;
        }

        generation.patchSettings({ referenceImages: result.referenceImages }, projectId);
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
    openWorkbenchWidget,
    projectId,
    queryClient,
    queries,
    supportedModels,
    t,
    vaeModels,
  ]);
};

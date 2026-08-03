import type { GalleryItemActionContext, GalleryItemActions } from '@features/gallery/react';
import type { VaeModelConfig } from '@features/generation/contracts';

import {
  galleryImages,
  galleryItemOrganization,
  galleryItems,
  galleryTransfers,
  toGalleryItemKey,
  toGalleryItemRef,
  type GalleryBoard,
  type GalleryImage,
  type GalleryImageMetadata,
  type GalleryItem,
  type GalleryItemMutationResult,
  type GalleryItemRef,
} from '@features/gallery';
import { getGalleryBoardLabel } from '@features/gallery/contracts';
import { invalidateGallery, patchGalleryItemCaches } from '@features/gallery/queries';
import { setPendingPromptTemplateDraft } from '@features/generation/react';
import { getMaxReferenceImages, isVaeModelConfig, isSupportedGenerateModel } from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { downloadBlob } from '@platform/browser/downloadBlob';
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
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { RequestDeletionConfirmation } from './useDeletionConfirmation';

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
export interface ImageActions extends GalleryItemActions {
  /** Whether the generate widget's current model can accept another reference image. */
  canUseAsReferenceImage: boolean;
  copyImage: (image: GalleryImage) => Promise<void>;
  deleteImages: (imageNames: string[]) => Promise<void>;
  /** Derives recall availability from already-fetched metadata and the current generate/model state. */
  deriveImageRecallCapabilities: (
    image: GalleryImage,
    metadata: GalleryImageMetadata | null
  ) => ImageRecallCapabilities;
  downloadImage: (image: GalleryImage) => Promise<void>;
  downloadImages: (imageNames: string[]) => Promise<void>;
  getImageRecallCapabilities: (image: GalleryImage, signal?: AbortSignal) => Promise<ImageRecallCapabilities>;
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
  getItemActionContext,
  onImagesDeleted,
  projectId,
  requestDeletionConfirmation,
}: {
  boards: GalleryBoard[];
  generateValues: Record<string, unknown>;
  getItemActionContext?: () => GalleryItemActionContext | null;
  projectId?: string;
  /** Called after a successful deletion so the host can select a neighboring image. */
  onImagesDeleted?: (imageNames: string[]) => void;
  requestDeletionConfirmation: RequestDeletionConfirmation;
}): ImageActions => {
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const commands = useWorkbenchCommands();
  const { gallery, generation, notifications } = commands;
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();
  const { t } = useTranslation();
  const confirmImageDeletion = useWorkbenchPreferenceSelector((preferences) => preferences.confirmImageDeletion);
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
    const getBoardName = (boardId: string) => {
      const board = boards.find((board) => board.id === boardId);

      return board ? getGalleryBoardLabel(board, t) : t('widgets.gallery.uncategorized');
    };
    const getLatestGenerateValues = () => {
      const snapshot = queries.getSnapshot();
      const project = projectId
        ? snapshot.projects.find((candidate) => candidate.id === projectId)
        : snapshot.activeProject;

      return project ? getProjectWidgetValues(project, 'generate') : {};
    };
    const reportMutationOutcome = (
      action: 'delete' | 'move' | 'star' | 'unstar',
      requestedCount: number,
      result: GalleryItemMutationResult,
      boardId?: string
    ) => {
      if (result.failed.length > 0) {
        recordError(
          new Error(
            t(`widgets.gallery.itemActions.${action}.${result.succeeded.length > 0 ? 'partial' : 'failure'}`, {
              board: boardId ? getBoardName(boardId) : undefined,
              count: requestedCount,
              failed: result.failed.length,
              succeeded: result.succeeded.length,
            })
          )
        );
        return;
      }

      recordSuccess(
        t(`widgets.gallery.itemActions.${action}.success`, {
          board: boardId ? getBoardName(boardId) : undefined,
          count: result.succeeded.length,
        })
      );
    };
    const runItemMutation = async ({
      action,
      applyConfirmed,
      boardId,
      mutate,
      requested,
    }: {
      action: 'delete' | 'move' | 'star' | 'unstar';
      applyConfirmed: (result: GalleryItemMutationResult, signal: AbortSignal) => Promise<void> | void;
      boardId?: string;
      mutate: (signal: AbortSignal) => Promise<GalleryItemMutationResult>;
      requested: GalleryItemRef[];
    }): Promise<void> => {
      const owner = captureAccountScope();
      let error: unknown = null;
      let result: GalleryItemMutationResult | null = null;

      try {
        result = await mutate(owner.signal);

        assertAccountScopeCurrent(owner);
        await applyConfirmed(result, owner.signal);
      } catch (caught: unknown) {
        error = caught;
      }

      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      try {
        await invalidateGallery(queryClient, owner);
      } catch (caught: unknown) {
        error ??= caught;
      }

      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      if (error || !result) {
        recordError(error ?? new Error('Gallery item mutation did not return a result.'));
        return;
      }

      reportMutationOutcome(action, requested.length, result, boardId);
    };
    const deleteItemsConfirmed = (items: GalleryItemRef[]): Promise<void> => {
      let deletionContext: GalleryItemActionContext | null = null;
      let orderedRefs: GalleryItemRef[] | null = null;
      const isDeletionContextCurrent = (): boolean => {
        if (!deletionContext || !getItemActionContext) {
          return false;
        }

        const current = getItemActionContext();

        return Boolean(
          current &&
          current.filterIdentity === deletionContext.filterIdentity &&
          current.selectedItemKey === deletionContext.selectedItemKey
        );
      };

      return runItemMutation({
        action: 'delete',
        applyConfirmed: async (result, signal) => {
          if (result.succeeded.length === 0) {
            return;
          }

          let successor: GalleryItem | null = null;
          const primaryKey = deletionContext?.selectedItemKey ?? null;
          const succeededKeys = new Set(result.succeeded.map(toGalleryItemKey));

          if (deletionContext && primaryKey && succeededKeys.has(primaryKey) && isDeletionContextCurrent()) {
            const refs = orderedRefs ?? deletionContext.items.map(toGalleryItemRef);
            const primaryIndex = refs.findIndex((ref) => toGalleryItemKey(ref) === primaryKey);
            const ineligibleKeys = new Set([
              ...result.succeeded.map(toGalleryItemKey),
              ...result.failed.map(toGalleryItemKey),
            ]);
            let successorRef: GalleryItemRef | null = null;

            if (primaryIndex >= 0) {
              for (let index = primaryIndex - 1; index >= 0; index -= 1) {
                const candidate = refs[index];

                if (candidate && !ineligibleKeys.has(toGalleryItemKey(candidate))) {
                  successorRef = candidate;
                  break;
                }
              }

              if (!successorRef) {
                for (let index = primaryIndex + 1; index < refs.length; index += 1) {
                  const candidate = refs[index];

                  if (candidate && !ineligibleKeys.has(toGalleryItemKey(candidate))) {
                    successorRef = candidate;
                    break;
                  }
                }
              }
            }

            if (successorRef) {
              successor =
                deletionContext.items.find((item) => toGalleryItemKey(item) === toGalleryItemKey(successorRef)) ?? null;

              if (!successor) {
                try {
                  successor = await galleryItems.resolve(successorRef, signal);
                } catch {
                  successor = null;
                }
              }

              signal.throwIfAborted();

              if (!isDeletionContextCurrent()) {
                successor = null;
              }
            }
          }

          patchGalleryItemCaches(queryClient, { kind: 'delete', result });
          gallery.removeItems(result.succeeded.map(toGalleryItemKey));
          if (successor) {
            const failedKeys = new Set(result.failed.map(toGalleryItemKey));
            const retainedFailedKeys = items
              .filter((item) => failedKeys.has(toGalleryItemKey(item)))
              .map(toGalleryItemKey);

            if (retainedFailedKeys.length > 0) {
              gallery.setItemMultiSelection([...retainedFailedKeys, toGalleryItemKey(successor)], successor, projectId);
            } else {
              gallery.selectItem(successor, projectId);
            }
          }
          onImagesDeleted?.(result.succeeded.filter((item) => item.kind === 'image').map((item) => item.name));
        },
        mutate: async (signal) => {
          deletionContext = getItemActionContext?.() ?? null;

          if (
            deletionContext?.selectedItemKey &&
            items.some((item) => toGalleryItemKey(item) === deletionContext?.selectedItemKey)
          ) {
            try {
              orderedRefs = await deletionContext.loadOrderedRefs(signal);
            } catch {
              orderedRefs = deletionContext.items.map(toGalleryItemRef);
            }
          }

          signal.throwIfAborted();
          return galleryItemOrganization.delete(items, signal);
        },
        requested: items,
      });
    };
    const deleteItems = (items: GalleryItemRef[]): Promise<void> =>
      confirmImageDeletion
        ? requestDeletionConfirmation(items, () => deleteItemsConfirmed(items))
        : deleteItemsConfirmed(items);
    const moveItemsToBoard = (items: GalleryItemRef[], boardId: string): Promise<void> =>
      runItemMutation({
        action: 'move',
        applyConfirmed: (result) => {
          if (result.succeeded.length === 0) {
            return;
          }

          patchGalleryItemCaches(queryClient, { boardId, kind: 'move', result });
          gallery.patchItems(result.succeeded.map(toGalleryItemKey), { boardId });
        },
        boardId,
        mutate: (signal) => galleryItemOrganization.moveToBoard(items, boardId, signal),
        requested: items,
      });
    const setItemsStarred = (items: GalleryItemRef[], starred: boolean): Promise<void> =>
      runItemMutation({
        action: starred ? 'star' : 'unstar',
        applyConfirmed: (result) => {
          if (result.succeeded.length === 0) {
            return;
          }

          patchGalleryItemCaches(queryClient, { kind: 'star', result, starred });
          gallery.patchItems(result.succeeded.map(toGalleryItemKey), { starred });
        },
        mutate: (signal) => galleryItemOrganization.setStarred(items, starred, signal),
        requested: items,
      });
    const fetchItemBlob = async (item: GalleryItem, signal: AbortSignal): Promise<Blob> => {
      const response = await fetch(item.fullUrl, { signal });

      if (!response.ok) {
        throw new Error(`Download failed with status ${response.status}.`);
      }

      return response.blob();
    };
    const downloadItem = async (item: GalleryItem): Promise<void> => {
      const owner = captureAccountScope();

      try {
        const blob = await fetchItemBlob(item, owner.signal);

        assertAccountScopeCurrent(owner);
        downloadBlob(blob, item.name);
      } catch (error: unknown) {
        if (isAccountScopeCurrent(owner)) {
          recordError(error);
        }
      }
    };
    const downloadItems = async (items: GalleryItemRef[], loadedItems: GalleryItem[] = []): Promise<void> => {
      const owner = captureAccountScope();
      const loadedByKey = new Map(loadedItems.map((item) => [toGalleryItemKey(item), item]));
      const imageNames = items.filter((item) => item.kind === 'image').map((item) => item.name);
      const videoRefs = items.filter((item) => item.kind === 'video');
      let failedCount = 0;
      let succeededCount = 0;

      if (imageNames.length > 0) {
        try {
          const { blob, fileName } = await galleryTransfers.downloadArchive({
            imageNames,
            signal: owner.signal,
          });

          assertAccountScopeCurrent(owner);
          downloadBlob(blob, fileName);
          succeededCount += imageNames.length;
        } catch {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          failedCount += imageNames.length;
        }
      }

      for (const ref of videoRefs) {
        try {
          const loadedItem = loadedByKey.get(toGalleryItemKey(ref));
          const item = loadedItem ?? (await galleryItems.resolve(ref, owner.signal));

          assertAccountScopeCurrent(owner);
          if (item.kind !== 'video') {
            throw new Error(`Resolved ${ref.name} as the wrong media kind.`);
          }

          const blob = await fetchItemBlob(item, owner.signal);

          assertAccountScopeCurrent(owner);
          downloadBlob(blob, item.name);
          succeededCount += 1;
        } catch {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          failedCount += 1;
        }
      }

      if (failedCount > 0) {
        recordError(
          new Error(
            t('widgets.gallery.itemActions.download.partial', {
              count: items.length,
              failed: failedCount,
              succeeded: succeededCount,
            })
          )
        );
      } else {
        recordSuccess(
          t('widgets.gallery.itemActions.download.success', {
            count: succeededCount,
          })
        );
      }
    };
    const deriveImageRecallCapabilities = (
      image: GalleryImage,
      metadata: GalleryImageMetadata | null
    ): ImageRecallCapabilities => {
      if (!currentGenerateValues) {
        return EMPTY_IMAGE_RECALL_CAPABILITIES;
      }

      return getImageRecallCapabilities({
        currentValues: currentGenerateValues,
        image,
        metadata,
        models,
        supportedModels,
        vaeModels,
      });
    };

    return {
      deleteItems,
      downloadItem,
      downloadItems,
      moveItemsToBoard,
      openItemInNewTab: (item) => {
        window.open(item.fullUrl, '_blank', 'noopener');
      },
      openItemInPreview: (item) => {
        gallery.selectItem(item, projectId);
        openWorkbenchWidget('preview', { preferredRegions: ['center'], requireCenterView: true });
      },
      setItemsStarred,
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
      deleteImages: (imageNames) => deleteItems(imageNames.map((name) => ({ kind: 'image', name }))),
      deriveImageRecallCapabilities,
      downloadImage: async (image) => {
        const owner = captureAccountScope();

        try {
          const response = await fetch(image.imageUrl, { signal: owner.signal });
          const blob = await response.blob();

          assertAccountScopeCurrent(owner);
          downloadBlob(blob, image.imageName);
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
      downloadImages: (imageNames) => downloadItems(imageNames.map((name) => ({ kind: 'image', name }))),
      getImageRecallCapabilities: async (image, signal) => {
        const owner = captureAccountScope();

        if (!currentGenerateValues) {
          return EMPTY_IMAGE_RECALL_CAPABILITIES;
        }

        try {
          const requestSignal = signal ? AbortSignal.any([signal, owner.signal]) : owner.signal;
          const metadata = await galleryImages.metadata(image.imageName, requestSignal);

          assertAccountScopeCurrent(owner);
          return deriveImageRecallCapabilities(image, metadata);
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
      moveImagesToBoard: (imageNames, boardId) =>
        moveItemsToBoard(
          imageNames.map((name) => ({ kind: 'image', name })),
          boardId
        ),
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
      setImagesStarred: (imageNames, starred) =>
        setItemsStarred(
          imageNames.map((name) => ({ kind: 'image', name })),
          starred
        ),
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
    confirmImageDeletion,
    currentGenerateValues,
    commands,
    gallery,
    generateValues,
    generation,
    getItemActionContext,
    models,
    notifications,
    onImagesDeleted,
    openWorkbenchWidget,
    projectId,
    queryClient,
    queries,
    requestDeletionConfirmation,
    supportedModels,
    t,
    vaeModels,
  ]);
};

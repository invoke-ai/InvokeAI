import type { GalleryItemActionContext, GalleryItemActions } from '@features/gallery/react';
import type { VaeModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { WorkbenchSnapshot } from '@workbench/workbenchStore';

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
  type GalleryItemKey,
  type GalleryItemMutationResult,
  type GalleryItemRef,
  galleryVideos,
  type GalleryVideoItem,
} from '@features/gallery';
import { getGalleryBoardLabel, getGalleryDeletionSuccessor } from '@features/gallery/contracts';
import {
  getGalleryItemBoardIdsFromCaches,
  getGalleryItemStarredFromCaches,
  invalidateGallery,
  patchGalleryItemCaches,
} from '@features/gallery/queries';
import { flushGenerateDrafts, setPendingPromptTemplateDraft } from '@features/generation/react';
import { getArchitectureCapabilitiesSnapshot, subscribeArchitectureCapabilities } from '@features/generation/runtime';
import { getMaxReferenceImages, isVaeModelConfig, isSupportedGenerateModel } from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { downloadBlob } from '@platform/browser/downloadBlob';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { useQueryClient } from '@tanstack/react-query';
import {
  createCanvasFromImages,
  getCanvasImportNotice,
  getCanvasEngine,
  importGalleryImagesToCanvas,
  type GalleryCanvasImportDestination,
} from '@workbench/canvas-operations/api';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { getProjectWidgetInstance, getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchQueries, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { RequestDeletionConfirmation } from './useDeletionConfirmation';

import { appendReferenceImage } from './appendReferenceImage';
import { recordCanvasImportError } from './canvasImportError';
import { executeImageRecall, executeLoadImageWorkflow, getCurrentGenerateValues } from './executeImageRecall';
import {
  appendReferenceVideo,
  canAppendReferenceVideo,
  executeVideoRecall,
  placeInitialVideo,
} from './executeVideoRecall';
import {
  captureGalleryWidgetKeyValues,
  collectGalleryStoreKnownItemFields,
  diffGalleryWidgetKeyValues,
  selectItemKeysUnchangedSince,
  selectRestorableGalleryWidgetPatches,
  type GalleryWidgetKeySnapshotEntry,
} from './galleryOptimisticRollback';
import {
  getMetadataPrompts,
  EMPTY_IMAGE_RECALL_CAPABILITIES,
  getImageRecallCapabilities,
  type ImageRecallCapabilities,
  type ImageRecallKind,
} from './imageRecall';
import {
  EMPTY_VIDEO_RECALL_CAPABILITIES,
  getVideoRecallCapabilities as deriveVideoRecallCapabilitiesFromMetadata,
  type VideoRecallCapabilities,
  type VideoRecallKind,
} from './videoRecall';

/**
 * Share backend image mutations across surfaces; patch Gallery caches optimistically and invalidate affected
 * server state.
 */
export interface ImageActions extends GalleryItemActions {
  /** Whether the generate widget's current model can accept another reference image. */
  canUseAsReferenceImage: boolean;
  /** Whether the video widget's current model takes reference videos and has room for another. */
  canUseAsReferenceVideo: boolean;
  copyImage: (image: GalleryImage) => Promise<void>;
  /** Opens a new project whose canvas holds these images as raster layers. */
  createCanvasFromImages: (images: readonly GalleryImage[]) => Promise<void>;
  deleteImages: (imageNames: string[]) => Promise<void>;
  /** Derives recall availability from already-fetched metadata and the current generate/model state. */
  deriveImageRecallCapabilities: (
    image: GalleryImage,
    metadata: GalleryImageMetadata | null
  ) => ImageRecallCapabilities;
  downloadImage: (image: GalleryImage) => Promise<void>;
  downloadImages: (imageNames: string[]) => Promise<void>;
  getImageRecallCapabilities: (image: GalleryImage, signal?: AbortSignal) => Promise<ImageRecallCapabilities>;
  /** Replaces the project graph with the workflow embedded in the image and opens the editor on it. */
  loadImageWorkflow: (image: GalleryImage) => Promise<void>;
  /** Recall availability for a gallery video, from its recorded core_metadata. */
  getVideoRecallCapabilities: (item: GalleryVideoItem, signal?: AbortSignal) => Promise<VideoRecallCapabilities>;
  moveImagesToBoard: (imageNames: string[], boardId: string) => Promise<void>;
  openImageInPreview: (image: GalleryImage) => void;
  recallImageData: (image: GalleryImage, kind: ImageRecallKind) => Promise<void>;
  /** Applies a gallery video's recorded parameters to the Video panel. */
  recallVideoData: (item: GalleryVideoItem, kind: VideoRecallKind) => Promise<void>;
  /** Sets a gallery video as the Video panel's Initial Video, as that field would. */
  sendToInitialVideo: (item: GalleryVideoItem) => void;
  /** Opens the generate widget's template editor prefilled from this image's prompts. */
  savePromptAsTemplate: (image: GalleryImage) => Promise<void>;
  selectForCompare: (image: GalleryImage) => void;
  sendToCanvas: (images: readonly GalleryImage[], destination: GalleryCanvasImportDestination) => Promise<void>;
  setImagesStarred: (imageNames: string[], starred: boolean) => Promise<void>;
  useAsReferenceImage: (image: GalleryImage) => void;
  useAsReferenceVideo: (item: GalleryVideoItem) => void;
}

const EMPTY_WIDGET_VALUES: Record<string, unknown> = {};

/**
 * Selected as a boolean so Video panel edits re-render action consumers only when the answer flips. The cache skips
 * re-normalizing video values that have not changed since the last store update.
 */
const createCanUseAsReferenceVideoSelector = (models: readonly ModelConfig[], projectId: string | undefined) => {
  let lastValues: Record<string, unknown> | null = null;
  let lastResult = false;

  return (snapshot: WorkbenchSnapshot): boolean => {
    const project = projectId
      ? snapshot.projects.find((candidate) => candidate.id === projectId)
      : snapshot.activeProject;
    const videoValues = (project && getProjectWidgetInstance(project, 'video')?.state.values) ?? EMPTY_WIDGET_VALUES;

    if (videoValues !== lastValues) {
      lastValues = videoValues;
      lastResult = canAppendReferenceVideo({ models, videoValues });
    }

    return lastResult;
  };
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
  // Both read architecture policy, so they are read inside the capability store's selector: memoised
  // on the project alone they would keep the answer given before the table arrived.
  const { canUseAsReferenceImage, currentGenerateValues } = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(() => {
      const values = getCurrentGenerateValues({ generateValues, supportedModels });

      return {
        canUseAsReferenceImage: Boolean(values && values.referenceImages.length < getMaxReferenceImages(values.model)),
        currentGenerateValues: values,
      };
    }, [generateValues, supportedModels])
  );

  const selectCanUseAsReferenceVideo = useMemo(
    () => createCanUseAsReferenceVideoSelector(models, projectId),
    [models, projectId]
  );
  const canUseAsReferenceVideo = useWorkbenchSelector(selectCanUseAsReferenceVideo, Object.is);

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
    const getLatestVideoValues = () => {
      const snapshot = queries.getSnapshot();
      const project = projectId
        ? snapshot.projects.find((candidate) => candidate.id === projectId)
        : snapshot.activeProject;

      return project ? getProjectWidgetValues(project, 'video') : {};
    };
    // Snapshot deletion-sensitive widget values across all projects; cache rollback cannot restore them. Diff
    // before/after values for conflict-safe restoration.
    const applyGalleryItemRemoval = (itemKeys: GalleryItemKey[]): GalleryWidgetKeySnapshotEntry[] => {
      const before = captureGalleryWidgetKeyValues(queries.getSnapshot().projects);

      gallery.removeItems(itemKeys);

      return diffGalleryWidgetKeyValues(before, queries.getSnapshot().projects);
    };
    const restoreGalleryItemRemoval = (entries: GalleryWidgetKeySnapshotEntry[]) => {
      const patches = selectRestorableGalleryWidgetPatches(entries, queries.getSnapshot().projects);

      for (const patch of patches) {
        // Restore exact prior values as a CAS-guarded system patch; widget patches are not undoable, and
        // user-origin patches would auto-route.
        commands.widgets.patchValues(patch.widgetId, patch.values, patch.projectId, 'system');
      }
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
      rollback,
    }: {
      action: 'delete' | 'move' | 'star' | 'unstar';
      applyConfirmed: (result: GalleryItemMutationResult, signal: AbortSignal) => Promise<void> | void;
      boardId?: string;
      mutate: (signal: AbortSignal) => Promise<GalleryItemMutationResult>;
      requested: GalleryItemRef[];
      /** Undo the optimistic apply. Runs once, only when the account scope that requested
       *  it is still current, before the (likely also-failing) trailing invalidation. */
      rollback?: () => void;
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
        // Roll back total failures immediately; an offline mutation often means invalidation also fails.
        if (isAccountScopeCurrent(owner)) {
          rollback?.();
        }
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
      // Capture successor context before optimistic removal. Partial failures reconcile via invalidation; total
      // failures restore widget snapshots directly.
      const deletionContext = getItemActionContext?.() ?? null;
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
      const rollbackCaches = patchGalleryItemCaches(queryClient, {
        kind: 'delete',
        result: { failed: [], succeeded: items },
      });
      // Guard cache rollback shared by partial/total failures so an exception after partial restoration cannot
      // roll it back twice.
      let cachesRolledBack = false;
      const rollbackCachesOnce = () => {
        if (cachesRolledBack) {
          return;
        }

        cachesRolledBack = true;
        rollbackCaches();
      };
      // Once backend-confirmed deletion starts applying, later callback failures must not restore deleted items.
      let confirmedApplied = false;
      const galleryWidgetSnapshot = applyGalleryItemRemoval(items.map(toGalleryItemKey));

      return runItemMutation({
        action: 'delete',
        applyConfirmed: async (result, signal) => {
          if (result.failed.length > 0) {
            rollbackCachesOnce();
          }

          if (result.succeeded.length === 0) {
            return;
          }

          let successor: GalleryItem | null = null;
          const primaryKey = deletionContext?.selectedItemKey ?? null;
          const succeededKeys = new Set(result.succeeded.map(toGalleryItemKey));

          if (deletionContext && primaryKey && succeededKeys.has(primaryKey) && isDeletionContextCurrent()) {
            const refs = orderedRefs ?? deletionContext.items.map(toGalleryItemRef);
            const ineligibleKeys = new Set([...succeededKeys, ...result.failed.map(toGalleryItemKey)]);
            const successorRef = getGalleryDeletionSuccessor(refs, primaryKey, ineligibleKeys);

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

          confirmedApplied = true;
          patchGalleryItemCaches(queryClient, { kind: 'delete', result });
          gallery.removeItems(result.succeeded.map(toGalleryItemKey));
          if (successor) {
            const failedKeys = new Set(result.failed.map(toGalleryItemKey));
            const retainedFailedKeys = items
              .filter((item) => failedKeys.has(toGalleryItemKey(item)))
              .map(toGalleryItemKey);
            // The successor was chosen from the host's own list, so it is
            // stamped the way the host would stamp it — in the window it
            // came from, not the grid's.
            const selectionPage = deletionContext?.getItemSelectionPage?.(successor);

            if (retainedFailedKeys.length > 0) {
              const itemKeys = [...retainedFailedKeys, toGalleryItemKey(successor)];

              if (selectionPage === undefined) {
                gallery.setItemMultiSelection(itemKeys, successor, projectId);
              } else {
                gallery.setItemMultiSelection(itemKeys, successor, projectId, selectionPage);
              }
            } else if (selectionPage === undefined) {
              gallery.selectItem(successor, projectId);
            } else {
              gallery.selectItem(successor, projectId, selectionPage, true);
            }
          }
          onImagesDeleted?.(result.succeeded.filter((item) => item.kind === 'image').map((item) => item.name));
        },
        mutate: async (signal) => {
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
        rollback: () => {
          if (confirmedApplied) {
            return;
          }

          rollbackCachesOnce();
          restoreGalleryItemRemoval(galleryWidgetSnapshot);
        },
      });
    };
    const deleteItems = (items: GalleryItemRef[]): Promise<void> =>
      confirmImageDeletion
        ? requestDeletionConfirmation(items, () => deleteItemsConfirmed(items))
        : deleteItemsConfirmed(items);
    const moveItemsToBoard = (items: GalleryItemRef[], boardId: string): Promise<void> => {
      // On partial move failure, restore then reapply confirmed items. Capture prior boards from cache and store,
      // preferring cache; invalidation reconciles conflicts.
      const previousBoardIds = new Map<GalleryItemKey, string>(
        [...collectGalleryStoreKnownItemFields(queries.getSnapshot().projects, items)].map(([key, fields]) => [
          key,
          fields.boardId,
        ])
      );

      for (const [key, cachedBoardId] of getGalleryItemBoardIdsFromCaches(queryClient, items)) {
        previousBoardIds.set(key, cachedBoardId);
      }

      const rollbackCaches = patchGalleryItemCaches(queryClient, {
        boardId,
        kind: 'move',
        result: { failed: [], succeeded: items },
      });
      // Partial and total failure paths share rollback; guard against applying it twice.
      let cachesRolledBack = false;
      const rollbackCachesOnce = () => {
        if (cachesRolledBack) {
          return;
        }

        cachesRolledBack = true;
        rollbackCaches();
      };
      // Restore store boards only while they still match this move's optimistic board; group by prior board to
      // preserve concurrent moves.
      const restorePreviousBoardIds = () => {
        const currentStoreBoardIds = collectGalleryStoreKnownItemFields(queries.getSnapshot().projects, items);
        const safeKeys = new Set(
          selectItemKeysUnchangedSince(
            [...previousBoardIds.keys()],
            boardId,
            (key) => currentStoreBoardIds.get(key)?.boardId
          )
        );
        const restoreByPreviousBoardId = new Map<string, GalleryItemKey[]>();

        for (const [key, previousBoardId] of previousBoardIds) {
          if (!safeKeys.has(key)) {
            continue;
          }

          const group = restoreByPreviousBoardId.get(previousBoardId) ?? [];

          group.push(key);
          restoreByPreviousBoardId.set(previousBoardId, group);
        }

        for (const [previousBoardId, keys] of restoreByPreviousBoardId) {
          gallery.patchItems(keys, { boardId: previousBoardId });
        }
      };

      gallery.patchItems(items.map(toGalleryItemKey), { boardId });

      return runItemMutation({
        action: 'move',
        applyConfirmed: (result) => {
          if (result.failed.length === 0) {
            return;
          }

          rollbackCachesOnce();
          patchGalleryItemCaches(queryClient, { boardId, kind: 'move', result });

          for (const ref of result.failed) {
            const key = toGalleryItemKey(ref);
            const previousBoardId = previousBoardIds.get(key);

            if (previousBoardId !== undefined) {
              gallery.patchItems([key], { boardId: previousBoardId });
            }
          }
        },
        boardId,
        mutate: (signal) => galleryItemOrganization.moveToBoard(items, boardId, signal),
        requested: items,
        rollback: () => {
          rollbackCachesOnce();
          restorePreviousBoardIds();
        },
      });
    };
    const patchStarredStoreOnly = (keys: GalleryItemKey[], starred: boolean): void => {
      if (keys.length === 0) {
        return;
      }

      gallery.patchItems(keys, { starred });
    };
    const setItemsStarred = (items: GalleryItemRef[], starred: boolean): Promise<void> => {
      // Cache star changes move items between listing partitions and need snapshot/CAS rollback. Store restoration
      // uses each actual prior flag, never blanket inversion.
      const previousStarred = new Map<GalleryItemKey, boolean>(
        [...collectGalleryStoreKnownItemFields(queries.getSnapshot().projects, items)].map(([key, fields]) => [
          key,
          fields.starred,
        ])
      );

      for (const [key, cachedStarred] of getGalleryItemStarredFromCaches(queryClient, items)) {
        previousStarred.set(key, cachedStarred);
      }

      let rollbackCaches: (() => void) | null = patchGalleryItemCaches(queryClient, {
        kind: 'star',
        result: { failed: [], succeeded: items },
        starred,
      });
      const rollbackCachesOnce = () => {
        rollbackCaches?.();
        rollbackCaches = null;
      };

      patchStarredStoreOnly(items.map(toGalleryItemKey), starred);

      return runItemMutation({
        action: starred ? 'star' : 'unstar',
        // Rejected refs must reappear where they were, so restore the
        // snapshot and re-apply only the confirmed ones.
        applyConfirmed: (result) => {
          if (result.failed.length === 0) {
            return;
          }

          rollbackCachesOnce();
          patchGalleryItemCaches(queryClient, {
            kind: 'star',
            result: { failed: [], succeeded: result.succeeded },
            starred,
          });
          patchStarredStoreOnly(result.failed.map(toGalleryItemKey), !starred);
        },
        mutate: (signal) => galleryItemOrganization.setStarred(items, starred, signal),
        requested: items,
        // Restore known prior flags only where this action's painted value remains; cache rollback restores its
        // own snapshot.
        rollback: () => {
          rollbackCachesOnce();

          const requestedKeys = items.map(toGalleryItemKey);
          const currentStoreFields = collectGalleryStoreKnownItemFields(queries.getSnapshot().projects, items);
          const safeStoreKeys = new Set(
            selectItemKeysUnchangedSince(requestedKeys, starred, (key) => currentStoreFields.get(key)?.starred)
          );
          const restoreGroups = new Map<boolean, GalleryItemKey[]>();

          for (const item of items) {
            const key = toGalleryItemKey(item);
            const priorStarred = previousStarred.get(key);

            if (priorStarred === undefined || !safeStoreKeys.has(key)) {
              continue;
            }

            const group = restoreGroups.get(priorStarred) ?? [];

            group.push(key);
            restoreGroups.set(priorStarred, group);
          }

          for (const [priorStarred, keys] of restoreGroups) {
            patchStarredStoreOnly(keys, priorStarred);
          }
        },
      });
    };
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
        const selectionPage = getItemActionContext?.()?.getItemSelectionPage?.(item);

        if (selectionPage === undefined) {
          gallery.selectItem(item, projectId);
        } else {
          gallery.selectItem(item, projectId, selectionPage, true);
        }
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
        const requestSignal = signal ? AbortSignal.any([signal, owner.signal]) : owner.signal;
        // Grid listings do not say whether an image embeds a workflow; images that
        // came through the record endpoints already do.
        const hasWorkflow =
          image.hasWorkflow !== undefined
            ? Promise.resolve(image.hasWorkflow)
            : galleryImages
                .resolve(image.imageName, requestSignal)
                .then((record) => record.hasWorkflow === true)
                .catch(() => false);

        if (!currentGenerateValues) {
          return { ...EMPTY_IMAGE_RECALL_CAPABILITIES, workflow: await hasWorkflow };
        }

        try {
          const [metadata, workflow] = await Promise.all([
            galleryImages.metadata(image.imageName, requestSignal),
            hasWorkflow,
          ]);

          assertAccountScopeCurrent(owner);
          return deriveImageRecallCapabilities({ ...image, hasWorkflow: workflow }, metadata);
        } catch {
          if (!isAccountScopeCurrent(owner)) {
            return EMPTY_IMAGE_RECALL_CAPABILITIES;
          }

          return {
            ...EMPTY_IMAGE_RECALL_CAPABILITIES,
            dimensions:
              Number.isFinite(image.width) && image.width >= 64 && Number.isFinite(image.height) && image.height >= 64,
            workflow: await hasWorkflow,
          };
        }
      },
      loadImageWorkflow: (image) =>
        executeLoadImageWorkflow({
          image,
          isProjectActive: () => !projectId || queries.isActiveProject(projectId),
          notifications,
          openWorkflowEditor: () =>
            openWorkbenchWidget('workflow', { preferredRegions: ['center'], requireCenterView: true }).ok,
          t,
        }),
      getVideoRecallCapabilities: async (item, signal) => {
        const owner = captureAccountScope();

        try {
          const requestSignal = signal ? AbortSignal.any([signal, owner.signal]) : owner.signal;
          const metadata = await galleryVideos.metadata(item.name, requestSignal);

          assertAccountScopeCurrent(owner);
          return deriveVideoRecallCapabilitiesFromMetadata(metadata);
        } catch {
          return EMPTY_VIDEO_RECALL_CAPABILITIES;
        }
      },
      recallVideoData: async (item, kind) => {
        const owner = captureAccountScope();
        const didRecall = await executeVideoRecall({
          commands,
          getVideoValues: getLatestVideoValues,
          item,
          kind,
          models,
          projectId,
        });

        // The value patch auto-routes invocation; the caller reveals the widget.
        if (isAccountScopeCurrent(owner) && didRecall && (!projectId || queries.isActiveProject(projectId))) {
          openWorkbenchWidget('video', { preferredRegions: ['left'] });
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
          t,
        });

        if (isAccountScopeCurrent(owner) && didRecall && (!projectId || queries.isActiveProject(projectId))) {
          openWorkbenchWidget('generate', { preferredRegions: ['left'] });
        }
      },
      selectForCompare: (image) => {
        gallery.setCompareImage(image, projectId);
      },
      createCanvasFromImages: async (images) => {
        const owner = captureAccountScope();
        flushGenerateDrafts();
        try {
          const result = await createCanvasFromImages({
            applyCanvasMutation: commands.canvas.apply,
            createProject: commands.projects.create,
            getProject: queries.getProject,
            images,
            isActiveProject: queries.isActiveProject,
          });

          assertAccountScopeCurrent(owner);
          if (result.status === 'imported' && result.failedImageNames.length === 0) {
            notifications.add({
              kind: 'success',
              title: t('widgets.canvas.import.newCanvasSuccess', { count: result.layerIds.length }),
            });
          } else {
            const notice = getCanvasImportNotice(result);
            notifications.add({ kind: notice.kind, title: t(notice.titleKey, notice.options ?? {}) });
          }
          if (result.status === 'imported' && result.projectId !== null && queries.isActiveProject(result.projectId)) {
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
      canUseAsReferenceImage,
      canUseAsReferenceVideo,
      sendToInitialVideo: (item) => {
        const placement = placeInitialVideo({ models, video: item, videoValues: getLatestVideoValues() });

        if (placement.status === 'full') {
          notifications.add({
            kind: 'info',
            message: t('widgets.video.placement.initialVideoFull'),
            title: t('widgets.video.referenceExtendCapFull'),
          });
          return;
        }

        openWorkbenchWidget('video', { preferredRegions: ['left'] });
        commands.widgets.patchValues('video', placement.patch, projectId);
        generation.setSource('video');
        if (!placement.usable) {
          notifications.add({
            kind: 'info',
            message: t('widgets.video.placement.initialVideoUnused'),
            title: t('widgets.video.placement.initialVideoSet'),
          });
        }
      },
      useAsReferenceVideo: (item) => {
        const placement = appendReferenceVideo({ models, video: item, videoValues: getLatestVideoValues() });

        // The menu offers this only when there is room; a race can still fill the last slot first.
        if (placement.status !== 'appended') {
          notifications.add({
            kind: 'info',
            message: t(
              placement.status === 'full'
                ? 'widgets.video.placement.referenceFull'
                : 'widgets.video.placement.referenceUnsupported'
            ),
            title: t('widgets.video.placement.referenceNotAdded'),
          });
          return;
        }

        commands.widgets.patchValues('video', placement.patch, projectId);
        openWorkbenchWidget('video', { preferredRegions: ['left'] });
      },
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
    canUseAsReferenceImage,
    canUseAsReferenceVideo,
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

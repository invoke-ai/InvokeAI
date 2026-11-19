import type { GalleryMediaUsageReport } from '@features/gallery/mediaUsage';
import type { GalleryItemActionContext, GalleryItemActions } from '@features/gallery/react';
import type { VaeModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { VideoConditioningRole } from '@features/video';
import type { WorkbenchSnapshot } from '@workbench/workbenchStore';

import {
  GALLERY_REQUEST_MAX_NAMES,
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
  type GalleryVideoItem,
} from '@features/gallery';
import {
  getGalleryAutoAddBoardId,
  getGalleryBoardLabel,
  getGalleryDeletionSuccessor,
  getPersistedSelectedGalleryItemKeys,
} from '@features/gallery/contracts';
import {
  canAddToCachedGalleryBoard,
  ensureGalleryBoardDestinations,
  fetchGalleryImageMetadata,
  getGalleryItemBoardIdsFromCaches,
  getGalleryItemStarredFromCaches,
  invalidateGallery,
  invalidateGalleryBoardDestinations,
  patchGalleryItemCaches,
} from '@features/gallery/queries';
import { setPendingPromptTemplateDraft } from '@features/generation/react';
import { getMaxReferenceImages, isVaeModelConfig, isSupportedGenerateModel } from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import {
  getArchitectureCapabilitiesSnapshot,
  subscribeArchitectureCapabilities,
} from '@features/models/architecturesStore';
import { submitUtilityGraph, type UtilityGraphSubmission, UtilityQueueError } from '@features/queue/utility';
import { copyImage } from '@platform/browser/clipboard';
import { downloadBlob } from '@platform/browser/downloadBlob';
import { createLogger } from '@platform/logging/logger';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { getApiErrorMessage } from '@platform/transport/http';
import { socketHub } from '@platform/transport/socketHub';
import { toaster } from '@platform/ui/toaster';
import { useQueryClient } from '@tanstack/react-query';
import {
  createCanvasFromImages,
  getCanvasImportNotice,
  getCanvasEngine,
  getSendAsControlNotice,
  importGalleryImagesToCanvas,
  sendGalleryImageAsControl,
  type GalleryCanvasImportDestination,
} from '@workbench/canvas-operations/api';
import { getProjectSyncSnapshot } from '@workbench/projects/syncStore';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useRevealLayerProperties } from '@workbench/widgets/layers/revealLayerProperties';
import { getProjectWidgetInstance, getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchQueries, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { MediaUsageHost } from './mediaUsageReport';
import type { RequestDeletionConfirmation } from './useDeletionConfirmation';
import type { VideoRecallKind } from './videoRecall';

import { appendReferenceImage } from './appendReferenceImage';
import { recordCanvasImportError } from './canvasImportError';
import { executeImageRecall, executeLoadMediaWorkflow, getCurrentGenerateValues } from './executeImageRecall';
import {
  appendReferenceVideo,
  executeVideoRecall,
  getVideoPlacementRoom,
  placeConditioningClip,
  placeInitialVideo,
  type VideoPlacementRoom,
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
import { buildPbrMapsGraph } from './pbrMapsGraph';

export interface ImageActions extends GalleryItemActions {
  canUseAsReferenceImage: boolean;
  canUseAsReferenceVideo: boolean;
  canUseAsConditioningClip: Record<VideoConditioningRole, boolean>;
  copyImage: (image: GalleryImage) => Promise<void>;
  createCanvasFromImages: (images: readonly GalleryImage[]) => Promise<void>;
  deleteImages: (imageNames: string[]) => Promise<void>;
  deriveImageRecallCapabilities: (
    image: GalleryImage,
    metadata: GalleryImageMetadata | null
  ) => ImageRecallCapabilities;
  downloadImage: (image: GalleryImage) => Promise<void>;
  downloadImages: (imageNames: string[]) => Promise<void>;
  generatePbrMaps: (image: GalleryImage) => Promise<void>;
  loadImageWorkflow: (image: GalleryImage) => Promise<void>;
  loadVideoWorkflow: (item: GalleryVideoItem) => Promise<void>;
  moveImagesToBoard: (imageNames: string[], boardId: string) => Promise<void>;
  openImageInPreview: (image: GalleryImage) => void;
  recallImageData: (image: GalleryImage, kind: ImageRecallKind) => Promise<void>;
  recallVideoData: (item: GalleryVideoItem, kind: VideoRecallKind) => Promise<void>;
  sendToInitialVideo: (item: GalleryVideoItem) => void;
  savePromptAsTemplate: (image: GalleryImage) => Promise<void>;
  selectForCompare: (image: GalleryImage) => void;
  sendToCanvas: (images: readonly GalleryImage[], destination: GalleryCanvasImportDestination) => Promise<void>;
  /** Fills the selected control layer when it has no image yet, else adds a control layer. */
  sendImageAsControl: (image: GalleryImage) => Promise<void>;
  setImagesStarred: (imageNames: string[], starred: boolean) => Promise<void>;
  useAsReferenceImage: (image: GalleryImage) => void;
  useAsReferenceVideo: (item: GalleryVideoItem) => void;
  /** Clears the media the clip excludes. */
  useAsConditioningClip: (item: GalleryVideoItem, role: VideoConditioningRole) => void;
}

const EMPTY_WIDGET_VALUES: Record<string, unknown> = {};
/** Per stage (server, then closed projects), before asking instead: the user is waiting on a click. */
const UNCONFIRMED_DELETION_CHECK_TIMEOUT_MS = 3_000;
const PENDING_CHECK_NOTICE_DELAY_MS = 400;

/** Items being checked, asked about or deleted, per account, shared by every deleting surface. */
const deletionsInFlight = new WeakMap<AccountScope, Set<GalleryItemKey>>();

const claimDeletion = (items: readonly GalleryItemRef[]): (() => void) | null => {
  const owner = captureAccountScope();
  const claimed = deletionsInFlight.get(owner) ?? new Set<GalleryItemKey>();
  const keys = items.map(toGalleryItemKey);

  if (keys.some((key) => claimed.has(key))) {
    return null;
  }

  deletionsInFlight.set(owner, claimed);
  keys.forEach((key) => claimed.add(key));

  return () => keys.forEach((key) => claimed.delete(key));
};

/** One notice for all concurrent silent-deletion checks, dismissed only when the last one ends. */
const PENDING_CHECK_NOTICE_ID = 'gallery-deletion-usage-check';
let pendingChecks = 0;

/** The release is safe to call more than once. */
const holdPendingCheckNotice = (title: string): (() => void) => {
  pendingChecks += 1;
  const timer = setTimeout(() => {
    toaster.create({ duration: Number.POSITIVE_INFINITY, id: PENDING_CHECK_NOTICE_ID, title, type: 'loading' });
  }, PENDING_CHECK_NOTICE_DELAY_MS);
  let released = false;

  return () => {
    if (released) {
      return;
    }
    released = true;
    clearTimeout(timer);
    pendingChecks -= 1;
    if (pendingChecks === 0) {
      toaster.dismiss(PENDING_CHECK_NOTICE_ID);
    }
  };
};

/** Booleans so Video panel edits re-render consumers only when an answer flips (the selector compares shallowly). */
const createVideoPlacementRoomSelector = (models: readonly ModelConfig[], projectId: string | undefined) => {
  let lastValues: Record<string, unknown> | null = null;
  let lastResult: VideoPlacementRoom = { conditioningAudio: false, conditioningVideo: false, referenceVideo: false };

  return (snapshot: WorkbenchSnapshot): VideoPlacementRoom => {
    const project = projectId
      ? snapshot.projects.find((candidate) => candidate.id === projectId)
      : snapshot.activeProject;
    const videoValues = (project && getProjectWidgetInstance(project, 'video')?.state.values) ?? EMPTY_WIDGET_VALUES;

    if (videoValues !== lastValues) {
      lastValues = videoValues;
      lastResult = getVideoPlacementRoom({ models, videoValues });
    }

    return lastResult;
  };
};

const toErrorMessage = (error: unknown): string =>
  getApiErrorMessage(error, error instanceof Error ? error.message : String(error));

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
  onImagesDeleted?: (imageNames: string[]) => void;
  requestDeletionConfirmation: RequestDeletionConfirmation;
}): ImageActions => {
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const revealLayerProperties = useRevealLayerProperties();
  const commands = useWorkbenchCommands();
  const { gallery, generation, notifications } = commands;
  const notify = useNotify();
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();
  const { t } = useTranslation();
  const confirmImageDeletion = useWorkbenchPreferenceSelector((preferences) => preferences.confirmImageDeletion);
  const models = useModelsSelector((snapshot) => snapshot.models);
  const supportedModels = useMemo(() => models.filter(isSupportedGenerateModel), [models]);
  const vaeModels = useMemo(() => models.filter(isVaeModelConfig).map((model) => model as VaeModelConfig), [models]);
  // Read inside the selector: memoised on the project alone they would keep the answer from before policy arrived.
  const { canUseAsReferenceImage, currentGenerateValues } = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(() => {
      const values = getCurrentGenerateValues({ generateValues, supportedModels });

      return {
        canUseAsReferenceImage: Boolean(
          values && values.referenceImages.length < getMaxReferenceImages(values.model, values)
        ),
        currentGenerateValues: values,
      };
    }, [generateValues, supportedModels])
  );

  const selectVideoPlacementRoom = useMemo(
    () => createVideoPlacementRoomSelector(models, projectId),
    [models, projectId]
  );
  const videoPlacementRoom = useWorkbenchSelector(selectVideoPlacementRoom);
  const canUseAsReferenceVideo = videoPlacementRoom.referenceVideo;
  const { conditioningAudio, conditioningVideo } = videoPlacementRoom;
  const canUseAsConditioningClip = useMemo(
    () => ({ audio: conditioningAudio, video: conditioningVideo }),
    [conditioningAudio, conditioningVideo]
  );

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const selectForCompare = useCallback(
    (image: GalleryImage) => gallery.setCompareImage(image, projectId),
    [gallery, projectId]
  );

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

      return board
        ? getGalleryBoardLabel(board, t)
        : t(boardId === 'none' ? 'widgets.gallery.uncategorized' : 'widgets.gallery.unknownBoard');
    };
    const getLatestProject = () => {
      const snapshot = queries.getSnapshot();

      return projectId ? snapshot.projects.find((candidate) => candidate.id === projectId) : snapshot.activeProject;
    };
    const getLatestWidgetValues = (widgetId: 'gallery' | 'generate' | 'video') => {
      const project = getLatestProject();

      return getResolvedProjectWidgetValues(project, widgetId, queries.getSnapshot().scopedDefaults);
    };
    const getLatestGenerateValues = () => getLatestWidgetValues('generate');
    const getLatestVideoValues = () => getLatestWidgetValues('video');
    // A host sees only loaded items, so it cannot tell a cleared selection from one moved to an unloaded item.
    const getPersistedGallerySelectionKey = (): string => {
      const values = getLatestWidgetValues('gallery');

      return JSON.stringify([
        typeof values.selectedImageName === 'string' ? values.selectedImageName : null,
        getPersistedSelectedGalleryItemKeys(values),
      ]);
    };
    // Cache rollback cannot restore deletion-sensitive widget values, so snapshot them across all projects.
    const applyGalleryItemRemoval = (itemKeys: GalleryItemKey[]): GalleryWidgetKeySnapshotEntry[] => {
      const before = captureGalleryWidgetKeyValues(queries.getSnapshot().projects);

      gallery.removeItems(itemKeys, { keepComparisonForSuccessor: true, projectId });

      return diffGalleryWidgetKeyValues(before, queries.getSnapshot().projects);
    };
    const restoreGalleryItemRemoval = (entries: GalleryWidgetKeySnapshotEntry[]) => {
      const patches = selectRestorableGalleryWidgetPatches(entries, queries.getSnapshot().projects);

      for (const patch of patches) {
        // CAS-guarded system patch: widget patches are not undoable, and user-origin patches would auto-route.
        commands.widgets.patchValues(patch.widgetId, patch.values, patch.projectId, 'system');
      }
    };
    const reportMutationOutcome = (
      action: 'delete' | 'move' | 'star' | 'unstar',
      requestedCount: number,
      result: GalleryItemMutationResult,
      boardId?: string,
      onUndo?: (() => void) | null
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

      const title = t(`widgets.gallery.itemActions.${action}.success`, {
        board: boardId ? getBoardName(boardId) : undefined,
        count: result.succeeded.length,
      });

      if (onUndo) {
        notify.undoable(title, { onUndo, projectId });
      } else {
        recordSuccess(title);
      }
    };
    const runItemMutation = async ({
      action,
      applyConfirmed,
      boardId,
      getUndo,
      mutate,
      requested,
      rollback,
    }: {
      action: 'delete' | 'move' | 'star' | 'unstar';
      applyConfirmed: (result: GalleryItemMutationResult, signal: AbortSignal) => Promise<void> | void;
      boardId?: string;
      getUndo?: (succeeded: readonly GalleryItemRef[]) => (() => void) | null;
      mutate: (signal: AbortSignal) => Promise<GalleryItemMutationResult>;
      requested: GalleryItemRef[];
      /** Runs once, only while the requesting account scope is current, before the trailing invalidation. */
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

      reportMutationOutcome(
        action,
        requested.length,
        result,
        boardId,
        result.failed.length === 0 ? getUndo?.(result.succeeded) : null
      );
    };
    const deleteItemsConfirmed = (items: GalleryItemRef[]): Promise<void> => {
      // Capture successor context before optimistic removal.
      const deletionContext = getItemActionContext?.() ?? null;
      let orderedRefs: GalleryItemRef[] | null = null;
      const isDeletionContextCurrent = (): boolean => {
        if (!deletionContext || !getItemActionContext) {
          return false;
        }

        const current = getItemActionContext();

        // Anything selected since this deletion's optimistic removal, loaded or not, makes the successor stale.
        return Boolean(
          current &&
          current.filterIdentity === deletionContext.filterIdentity &&
          (current.selectedItemKey === deletionContext.selectedItemKey ||
            getPersistedGallerySelectionKey() === selectionAfterRemoval)
        );
      };
      const rollbackCaches = patchGalleryItemCaches(queryClient, {
        kind: 'delete',
        result: { failed: [], succeeded: items },
      });
      // Shared by partial and total failures so cache rollback cannot run twice.
      let cachesRolledBack = false;
      const rollbackCachesOnce = () => {
        if (cachesRolledBack) {
          return;
        }

        cachesRolledBack = true;
        rollbackCaches();
      };
      // Once confirmed deletion starts applying, later callback failures must not restore deleted items.
      let confirmedApplied = false;
      const galleryWidgetSnapshot = applyGalleryItemRemoval(items.map(toGalleryItemKey));
      let selectionAfterRemoval = getPersistedGallerySelectionKey();

      return runItemMutation({
        action: 'delete',
        applyConfirmed: async (result, signal) => {
          // Rejected items remain, so restore what the optimistic removal cleared; confirmed removals reapply below.
          if (result.failed.length > 0) {
            // Adopt the restored selection only when it was still this deletion's, so a choice made meanwhile wins.
            const wasCurrent = isDeletionContextCurrent();

            rollbackCachesOnce();
            restoreGalleryItemRemoval(galleryWidgetSnapshot);

            if (wasCurrent) {
              selectionAfterRemoval = getPersistedGallerySelectionKey();
            }
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
          // Without a rollback the optimistic patch already lowered page totals; only pages refetched since count them.
          patchGalleryItemCaches(
            queryClient,
            { kind: 'delete', result },
            { totals: cachesRolledBack ? 'listing' : 'holder' }
          );
          gallery.removeItems(result.succeeded.map(toGalleryItemKey), {
            keepComparisonForSuccessor: successor !== null,
            projectId,
          });
          if (successor) {
            const failedKeys = new Set(result.failed.map(toGalleryItemKey));
            const retainedFailedKeys = items
              .filter((item) => failedKeys.has(toGalleryItemKey(item)))
              .map(toGalleryItemKey);
            // Stamp the successor in the window it came from, as the host would, not the grid's.
            const selectionPage = deletionContext?.getItemSelectionPage?.(successor);

            if (retainedFailedKeys.length > 0) {
              const itemKeys = [...retainedFailedKeys, toGalleryItemKey(successor)];

              if (selectionPage === undefined) {
                gallery.setItemMultiSelection(itemKeys, successor, projectId);
              } else {
                gallery.setItemMultiSelection(itemKeys, successor, projectId, selectionPage, true);
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
    const deleteItems: ImageActions['deleteItems'] = async (items, options) => {
      // Drop a repeated Delete for claimed items: running it again would delete twice and report a refused failure.
      const release = claimDeletion(items);

      if (!release) {
        return;
      }

      const usageHost: MediaUsageHost = {
        getLoadedProjects: () => queries.getSnapshot().projects,
        getSyncBase: (targetProjectId) => getProjectSyncSnapshot().projects[targetProjectId],
        openProjectId: projectId ?? queries.getSnapshot().activeProject.id,
      };
      const request = { kind: 'items', refs: items } as const;
      const checkUsage = (signal: AbortSignal) =>
        import('./mediaUsageReport').then(({ resolveMediaUsage }) => resolveMediaUsage(request, usageHost, { signal }));
      // Only one visible item may skip confirmation: a batch or off-screen delete always names what goes.
      const canSkipFutureConfirmations = items.length === 1 && options?.targetsOffScreen !== true;
      const confirm = (usage?: GalleryMediaUsageReport) =>
        requestDeletionConfirmation(items, () => deleteItemsConfirmed(items), {
          canSkipFutureConfirmations,
          checkUsage,
          returnFocus: options?.returnFocus,
          usage,
        });

      try {
        if (confirmImageDeletion || !canSkipFutureConfirmations) {
          return await confirm();
        }

        // Without routine confirmation one visible item goes silently only once the check finds it unused in time.
        const owner = captureAccountScope();
        const releaseNotice = holdPendingCheckNotice(t('widgets.gallery.deleteInUse.checking'));
        let usage: GalleryMediaUsageReport | undefined;
        let isUnused = false;
        let checkFailed = false;

        try {
          const [{ isUsedInLoadedProjects, resolveMediaUsage }, { isGalleryMediaUnused }] = await Promise.all([
            import('./mediaUsageReport'),
            import('@features/gallery/mediaUsage'),
          ]);
          const report = await resolveMediaUsage(request, usageHost, {
            projectReadsTimeoutMs: UNCONFIRMED_DELETION_CHECK_TIMEOUT_MS,
            serverTimeoutMs: UNCONFIRMED_DELETION_CHECK_TIMEOUT_MS,
          });

          if (!isGalleryMediaUnused(report)) {
            usage = report;
          } else {
            // The answer can be seconds old: a use added in this tab since still asks.
            isUnused = !isUsedInLoadedProjects(items, usageHost);
          }
        } catch {
          checkFailed = true;
        } finally {
          releaseNotice();
        }

        if (checkFailed) {
          // The resolver rejects only on an account change; the deletion belonged to the previous account.
          return isAccountScopeCurrent(owner) ? await confirm() : undefined;
        }

        return isUnused ? await deleteItemsConfirmed(items) : await confirm(usage);
      } finally {
        release();
      }
    };
    const groupByPrior = <Value>(
      refs: readonly GalleryItemRef[],
      getPrior: (key: GalleryItemKey) => Value | undefined
    ): Map<Value, GalleryItemRef[]> => {
      const groups = new Map<Value, GalleryItemRef[]>();

      for (const ref of refs) {
        const prior = getPrior(toGalleryItemKey(ref));

        if (prior === undefined) {
          continue;
        }

        const group = groups.get(prior);

        if (group) {
          group.push(ref);
        } else {
          groups.set(prior, [ref]);
        }
      }

      return groups;
    };
    // Prefer cache over store; invalidation reconciles conflicts.
    const readKnownBoardIds = (refs: readonly GalleryItemRef[]): Map<GalleryItemKey, string> => {
      const boardIds = new Map<GalleryItemKey, string>(
        [...collectGalleryStoreKnownItemFields(queries.getSnapshot().projects, refs)].map(([key, fields]) => [
          key,
          fields.boardId,
        ])
      );

      for (const [key, cachedBoardId] of getGalleryItemBoardIdsFromCaches(queryClient, refs)) {
        boardIds.set(key, cachedBoardId);
      }

      return boardIds;
    };
    const readKnownStarred = (refs: readonly GalleryItemRef[]): Map<GalleryItemKey, boolean> => {
      const starred = new Map<GalleryItemKey, boolean>(
        [...collectGalleryStoreKnownItemFields(queries.getSnapshot().projects, refs)].map(([key, fields]) => [
          key,
          fields.starred,
        ])
      );

      for (const [key, cachedStarred] of getGalleryItemStarredFromCaches(queryClient, refs)) {
        starred.set(key, cachedStarred);
      }

      return starred;
    };
    /** Undo reverts only items still holding this action's value, so a later move or star wins. */
    const getGalleryItemUndo = <Value>({
      getCurrent,
      painted,
      previous,
      revert,
      succeeded,
    }: {
      getCurrent: (refs: readonly GalleryItemRef[]) => ReadonlyMap<GalleryItemKey, Value>;
      painted: Value;
      previous: ReadonlyMap<GalleryItemKey, Value>;
      revert: (refs: GalleryItemRef[], prior: Value) => void;
      succeeded: readonly GalleryItemRef[];
    }): (() => void) | null => {
      const changed = succeeded.filter((ref) => previous.get(toGalleryItemKey(ref)) !== painted);
      const groups = groupByPrior(changed, (key) => previous.get(key));

      if (groups.size === 0) {
        return null;
      }

      return () => {
        const restorable = [...groups.values()].flat();
        const current = getCurrent(restorable);
        const unchanged = new Set(
          selectItemKeysUnchangedSince(restorable.map(toGalleryItemKey), painted, (key) => current.get(key))
        );
        let restored = 0;

        for (const [prior, refs] of groups) {
          const kept = refs.filter((ref) => unchanged.has(toGalleryItemKey(ref)));

          if (kept.length > 0) {
            restored += kept.length;
            revert(kept, prior);
          }
        }

        if (restored < changed.length) {
          notifications.add({
            kind: 'info',
            title:
              restored === 0
                ? t('widgets.gallery.itemActions.undoStale.none', { count: changed.length })
                : t('widgets.gallery.itemActions.undoStale.partial', { count: changed.length, restored }),
          });
        }
      };
    };
    const moveItemsToBoard = (items: GalleryItemRef[], boardId: string, offerUndo = true): Promise<void> => {
      const previousBoardIds = readKnownBoardIds(items);

      const rollbackCaches = patchGalleryItemCaches(queryClient, {
        boardId,
        kind: 'move',
        result: { failed: [], succeeded: items },
      });
      let cachesRolledBack = false;
      const rollbackCachesOnce = () => {
        if (cachesRolledBack) {
          return;
        }

        cachesRolledBack = true;
        rollbackCaches();
      };
      // Restore only boards still at this move's optimistic value, grouped by prior board, so concurrent moves survive.
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
        getUndo: (succeeded) =>
          offerUndo
            ? getGalleryItemUndo({
                getCurrent: readKnownBoardIds,
                painted: boardId,
                previous: previousBoardIds,
                revert: (refs, previousBoardId) => void moveItemsToBoard(refs, previousBoardId, false),
                succeeded,
              })
            : null,
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
    const setItemsStarred = (items: GalleryItemRef[], starred: boolean, offerUndo = true): Promise<void> => {
      // Star changes move items between listing partitions; restore each actual prior flag, never a blanket inversion.
      const previousStarred = readKnownStarred(items);

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
        getUndo: (succeeded) =>
          offerUndo
            ? getGalleryItemUndo({
                getCurrent: readKnownStarred,
                painted: starred,
                previous: previousStarred,
                revert: (refs, prior) => void setItemsStarred(refs, prior, false),
                succeeded,
              })
            : null,
        mutate: (signal) => galleryItemOrganization.setStarred(items, starred, signal),
        requested: items,
        // Restore prior flags only where this action's painted value remains.
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
      const archiveCount = Math.ceil(imageNames.length / GALLERY_REQUEST_MAX_NAMES);
      let failedCount = 0;
      let savedFileCount = 0;
      let succeededCount = 0;

      if (archiveCount > 1) {
        notifications.add({
          kind: 'info',
          message: t('widgets.gallery.itemActions.download.preparingArchives', { count: archiveCount }),
          title: t('widgets.gallery.preparingDownload'),
        });
      }

      for (let offset = 0; offset < imageNames.length; offset += GALLERY_REQUEST_MAX_NAMES) {
        const archiveNames = imageNames.slice(offset, offset + GALLERY_REQUEST_MAX_NAMES);

        try {
          const { blob, fileName } = await galleryTransfers.downloadArchive({
            imageNames: archiveNames,
            signal: owner.signal,
          });

          assertAccountScopeCurrent(owner);
          downloadBlob(blob, fileName);
          savedFileCount += 1;
          succeededCount += archiveNames.length;
        } catch {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          failedCount += archiveNames.length;
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
          savedFileCount += 1;
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
          archiveCount > 1
            ? t('widgets.gallery.itemActions.download.successArchives', { count: savedFileCount })
            : t('widgets.gallery.itemActions.download.success', { count: succeededCount })
        );
      }
    };
    const deriveImageRecallCapabilities = (
      image: GalleryImage,
      metadata: GalleryImageMetadata | null
    ): ImageRecallCapabilities => {
      if (!currentGenerateValues) {
        return { ...EMPTY_IMAGE_RECALL_CAPABILITIES, workflow: image.hasWorkflow === true };
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
          await copyImage(blob);
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
      generatePbrMaps: async (image) => {
        const owner = captureAccountScope();
        const galleryValues = getLatestWidgetValues('gallery');
        let submission: UtilityGraphSubmission;
        const reportFailure = (error: unknown) => {
          const message = toErrorMessage(error);

          createLogger({ area: 'image-actions', namespace: 'gallery', projectId }, { owner }).error({ error, message });
          notifications.add({ kind: 'error', message, title: t('widgets.gallery.itemActions.pbrMaps.failed') });
        };

        try {
          const pendingBoards = ensureGalleryBoardDestinations(queryClient);

          if (pendingBoards) {
            await pendingBoards;
          }
          assertAccountScopeCurrent(owner);
          const boardId = getGalleryAutoAddBoardId(galleryValues, (id) => canAddToCachedGalleryBoard(id, queryClient));

          // Ahead of queued generations, as the legacy action was; the account signal ends the watch.
          submission = await submitUtilityGraph({
            graph: buildPbrMapsGraph({ boardId, imageName: image.imageName }),
            hub: socketHub,
            prepend: true,
            signal: owner.signal,
          });
        } catch (error: unknown) {
          if (isAccountScopeCurrent(owner)) {
            void invalidateGalleryBoardDestinations(queryClient, owner);
            reportFailure(error);
          }
          return;
        }

        // Attach settlement before anything can throw, so an aborted watch never rejects unobserved.
        void submission.settled.then(
          () => {
            if (!isAccountScopeCurrent(owner)) {
              return;
            }

            void invalidateGallery(queryClient, owner);
            recordSuccess(
              t('widgets.gallery.itemActions.pbrMaps.completed'),
              t('widgets.gallery.itemActions.pbrMaps.completedMessage', { name: image.imageName })
            );
          },
          (error: unknown) => {
            if (!isAccountScopeCurrent(owner)) {
              return;
            }
            if (error instanceof UtilityQueueError && error.reason === 'canceled') {
              notifications.add({
                kind: 'info',
                message: t('widgets.gallery.itemActions.pbrMaps.canceledMessage', { name: image.imageName }),
                title: t('widgets.gallery.itemActions.pbrMaps.canceled'),
              });
              return;
            }
            // Gone without a terminal event (cleared or pruned while offline), so the gallery may hold its maps.
            if (error instanceof UtilityQueueError && error.reason === 'removed') {
              void invalidateGallery(queryClient, owner);
              notifications.add({
                kind: 'info',
                message: t('widgets.gallery.itemActions.pbrMaps.removedMessage', { name: image.imageName }),
                title: t('widgets.gallery.itemActions.pbrMaps.removed'),
              });
              return;
            }
            // An abort only ends the watch; the backend item keeps running.
            if (!(error instanceof UtilityQueueError && error.reason === 'aborted')) {
              void invalidateGalleryBoardDestinations(queryClient, owner);
              reportFailure(error);
            }
          }
        );

        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        recordSuccess(
          t('widgets.gallery.itemActions.pbrMaps.queued'),
          t('widgets.gallery.itemActions.pbrMaps.queuedMessage', { name: image.imageName })
        );
      },
      loadImageWorkflow: (image) =>
        executeLoadMediaWorkflow({
          isProjectActive: () => !projectId || queries.isActiveProject(projectId),
          media: { kind: 'image', name: image.imageName },
          notifications,
          openWorkflowEditor: () =>
            openWorkbenchWidget('workflow', { preferredRegions: ['center'], requireCenterView: true }).ok,
          t,
        }),
      loadVideoWorkflow: (item) =>
        executeLoadMediaWorkflow({
          isProjectActive: () => !projectId || queries.isActiveProject(projectId),
          media: { kind: 'video', name: item.name },
          notifications,
          openWorkflowEditor: () =>
            openWorkbenchWidget('workflow', { preferredRegions: ['center'], requireCenterView: true }).ok,
          t,
        }),
      recallVideoData: async (item, kind) => {
        const owner = captureAccountScope();
        const didRecall = await executeVideoRecall({
          commands,
          getVideoValues: getLatestVideoValues,
          item,
          kind,
          models,
          projectId,
          queryClient,
          t,
        });

        if (isAccountScopeCurrent(owner) && didRecall && (!projectId || queries.isActiveProject(projectId))) {
          openWorkbenchWidget('video', { preferredRegions: ['left'] });
        }
      },
      savePromptAsTemplate: async (image) => {
        const owner = captureAccountScope();

        try {
          const metadata = await fetchGalleryImageMetadata(queryClient, owner, image.imageName);

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
          canvasTarget: projectId
            ? {
                commands,
                projectId,
                getProject: () => (queries.isActiveProject(projectId) ? (getLatestProject() ?? null) : null),
              }
            : undefined,
          getGenerateValues: getLatestGenerateValues,
          image,
          isCanvasFrame: getLatestProject()?.invocation.sourceId === 'canvas',
          kind,
          models,
          projectId,
          queryClient,
          t,
        });

        if (isAccountScopeCurrent(owner) && didRecall && (!projectId || queries.isActiveProject(projectId))) {
          openWorkbenchWidget('generate', { preferredRegions: ['left'] });
        }
      },
      selectForCompare,
      createCanvasFromImages: async (images) => {
        const owner = captureAccountScope();
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
              title: t('widgets.canvas.import.newProjectSuccess', { count: result.layerIds.length }),
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
      sendImageAsControl: async (image) => {
        const owner = captureAccountScope();

        try {
          const targetProjectId = projectId ?? queries.getSnapshot().activeProject.id;
          const project = queries.getProject(targetProjectId);
          const result = project
            ? await sendGalleryImageAsControl({
                applyCanvasMutation: commands.canvas.apply,
                engine: getCanvasEngine(project.id) ?? null,
                getProject: queries.getProject,
                image,
                isActiveProject: queries.isActiveProject,
                project,
              })
            : ({ status: 'stale-project' } as const);

          assertAccountScopeCurrent(owner);
          const notice = getSendAsControlNotice(result);
          notifications.add({ kind: notice.kind, title: t(notice.titleKey, notice.options ?? {}) });

          if ((result.status === 'filled' || result.status === 'created') && queries.isActiveProject(targetProjectId)) {
            // No trigger to return focus to, so the Canvas takes it.
            openWorkbenchWidget('canvas', { preferredRegions: ['center'], requireCenterView: true });
            revealLayerProperties(result.layerId);
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
      canUseAsConditioningClip,
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

        // A race can still fill the last slot first.
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
      useAsConditioningClip: (item, role) => {
        const placement = placeConditioningClip({ models, role, video: item, videoValues: getLatestVideoValues() });

        // The panel's model can still change first.
        if (placement.status !== 'placed') {
          notifications.add({
            kind: 'info',
            message: t(
              placement.status === 'no-picture'
                ? 'widgets.video.placement.conditioningClipNoPicture'
                : 'widgets.video.placement.conditioningClipUnsupported'
            ),
            title: t('widgets.video.placement.conditioningClipNotSet'),
          });
          return;
        }

        commands.widgets.patchValues('video', placement.patch, projectId);
        openWorkbenchWidget('video', { preferredRegions: ['left'] });
        if (placement.displaced) {
          notifications.add({
            kind: 'info',
            message: t('widgets.video.placement.conditioningClipDisplaced'),
            title: t('widgets.video.placement.conditioningClipSet'),
          });
        }
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
    canUseAsConditioningClip,
    canUseAsReferenceImage,
    canUseAsReferenceVideo,
    confirmImageDeletion,
    currentGenerateValues,
    commands,
    gallery,
    generation,
    getItemActionContext,
    models,
    notifications,
    notify,
    onImagesDeleted,
    openWorkbenchWidget,
    projectId,
    queryClient,
    queries,
    requestDeletionConfirmation,
    revealLayerProperties,
    selectForCompare,
    supportedModels,
    t,
    vaeModels,
  ]);
};

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
  type GalleryItemKey,
  type GalleryItemMutationResult,
  type GalleryItemRef,
} from '@features/gallery';
import { getGalleryBoardLabel } from '@features/gallery/contracts';
import {
  getGalleryItemBoardIdsFromCaches,
  getGalleryItemStarredFromCaches,
  invalidateGallery,
  patchGalleryItemCaches,
} from '@features/gallery/queries';
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
    // A deleted item can be visible only through a project's `gallery` widget
    // values (e.g. the `recentImages` overlay for a just-generated image that
    // hasn't reached the backend query cache yet) or the `upscale` widget's
    // locked input, none of which `patchGalleryItemCaches` can restore.
    // Snapshot the fields `removeGalleryItems` can touch, across every open
    // project (images aren't project-scoped), diffed before/after so the
    // restore can apply the same "still current" conflict rule
    // `patchGalleryItemCaches`'s own rollback uses instead of clobbering
    // whatever a concurrent generation or selection wrote in the meantime.
    const applyGalleryItemRemoval = (itemKeys: GalleryItemKey[]): GalleryWidgetKeySnapshotEntry[] => {
      const before = captureGalleryWidgetKeyValues(queries.getSnapshot().projects);

      gallery.removeItems(itemKeys);

      return diffGalleryWidgetKeyValues(before, queries.getSnapshot().projects);
    };
    const restoreGalleryItemRemoval = (entries: GalleryWidgetKeySnapshotEntry[]) => {
      const patches = selectRestorableGalleryWidgetPatches(entries, queries.getSnapshot().projects);

      for (const patch of patches) {
        // `patchWidgetValues` is documented "not undoable" (workbenchState.ts);
        // this restore works around that by re-applying the exact prior values
        // as a forward patch, guarded by the CAS check above. `origin: 'system'`
        // keeps it from tripping the auto-route side effects a user-driven
        // widget patch would trigger.
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
        // The whole mutation failed — nothing was confirmed, so the optimistic
        // apply must not stand. The trailing invalidation cannot be relied on
        // here: whatever killed the mutation (offline) usually kills it too.
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
      // Optimistic: the items vanish immediately. Capture the action context
      // first — successor selection reasons about the pre-removal item list —
      // and keep a snapshot rollback so a failed delete resurrects the lists
      // before the confirmed subset is re-applied. On a partial failure, the
      // trailing invalidation is left to bring a failed ref's recent-image
      // overlay entry back from the backend list; on a total failure it can't
      // be relied on, so the gallery widget snapshot restores it directly.
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
      // `rollbackCaches` is invoked from two independent places below (the
      // partial-failure branch inside `applyConfirmed`, and the total-failure
      // `rollback`): guard so an `applyConfirmed` that throws after already
      // rolling back a partial failure can't undo the cache patch twice.
      let cachesRolledBack = false;
      const rollbackCachesOnce = () => {
        if (cachesRolledBack) {
          return;
        }

        cachesRolledBack = true;
        rollbackCaches();
      };
      // Once `applyConfirmed` starts applying a backend-confirmed result (some
      // items really were deleted), nothing after that point may trigger a
      // full rollback even if it throws — `onImagesDeleted` is a caller-
      // supplied callback invoked after confirmation and can throw for
      // reasons that have nothing to do with the mutation itself.
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

          confirmedApplied = true;
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
      // Optimistic: the items leave the current board view immediately. A
      // vanished item cannot be restored by another patch, so the failure
      // path rolls the lists back wholesale and re-applies the confirmed
      // subset; the trailing invalidation reconciles anything the rollback
      // had to skip as conflicted. `getGalleryItemBoardIdsFromCaches` only
      // sees items a list query has already fetched, so a just-generated
      // image known only through the recent-image overlay would otherwise
      // have no prior board to restore — read that from the store too, cache
      // taking precedence where both know the item.
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
      // See the delete path's `cachesRolledBack` note: `rollbackCaches` is
      // reachable both from the partial-failure branch below and from the
      // total-failure `rollback`, so guard against undoing it twice.
      let cachesRolledBack = false;
      const rollbackCachesOnce = () => {
        if (cachesRolledBack) {
          return;
        }

        cachesRolledBack = true;
        rollbackCaches();
      };
      // The cache rollback above is already CAS-guarded per query
      // (`patchGalleryItemCaches`'s own `if (current === after)` rule), but
      // the store patch below is a separate, unconditional write: a second
      // move that painted this item onto yet another board while the first
      // request was still hanging offline must not get clobbered back to the
      // first move's prior board. Only restore items still on the board
      // *this* move painted them onto, and group survivors by prior board id
      // to dispatch one store patch per group instead of one per item.
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
    const patchItemsStarred = (refs: GalleryItemRef[], starred: boolean): void => {
      if (refs.length === 0) {
        return;
      }

      patchGalleryItemCaches(queryClient, { kind: 'star', result: { failed: [], succeeded: refs }, starred });
      gallery.patchItems(refs.map(toGalleryItemKey), { starred });
    };
    // Split cache/store writers used only by the CAS-guarded rollback below,
    // where the two sides can pass or fail the "still current" check
    // independently (e.g. a trailing invalidation already reconciled the
    // cache but the store overlay wasn't touched).
    const patchStarredCacheOnly = (refs: GalleryItemRef[], starred: boolean): void => {
      if (refs.length === 0) {
        return;
      }

      patchGalleryItemCaches(queryClient, { kind: 'star', result: { failed: [], succeeded: refs }, starred });
    };
    const patchStarredStoreOnly = (keys: GalleryItemKey[], starred: boolean): void => {
      if (keys.length === 0) {
        return;
      }

      gallery.patchItems(keys, { starred });
    };
    const setItemsStarred = (items: GalleryItemRef[], starred: boolean): Promise<void> => {
      // Optimistic: a star toggle's outcome is the request itself except for
      // the rare rejected ref, so paint the whole selection immediately and
      // flip back only what the backend refuses. The trailing gallery
      // invalidation reconciles either path with the server. A total failure
      // can't rely on that invalidation, so capture each item's actual prior
      // flag up front (cache, falling back to the store overlay for items a
      // list query hasn't fetched) — blanket-inverting the whole batch would
      // wrongly flip items that already matched the target flag beforehand.
      const previousStarred = new Map<GalleryItemKey, boolean>(
        [...collectGalleryStoreKnownItemFields(queries.getSnapshot().projects, items)].map(([key, fields]) => [
          key,
          fields.starred,
        ])
      );

      for (const [key, cachedStarred] of getGalleryItemStarredFromCaches(queryClient, items)) {
        previousStarred.set(key, cachedStarred);
      }

      patchItemsStarred(items, starred);

      return runItemMutation({
        action: starred ? 'star' : 'unstar',
        applyConfirmed: (result) => patchItemsStarred(result.failed, !starred),
        mutate: (signal) => galleryItemOrganization.setStarred(items, starred, signal),
        requested: items,
        // Total failure: nothing was confirmed. Restore each item to its
        // actual prior flag rather than blanket-inverting the batch. An item
        // with no known prior flag is left as painted — there's nothing safe
        // to revert it to. Unlike the cache half of delete/move,
        // `patchGalleryItemCaches` here is a fresh forward patch with no
        // built-in CAS of its own (star's optimistic apply is a value flip,
        // not a snapshot/restore pair) — so both the cache and the store
        // writes need their own "still painted, nothing moved on" check
        // before a concurrent single-item toggle gets clobbered.
        rollback: () => {
          const requestedKeys = items.map(toGalleryItemKey);
          const currentCacheStarred = getGalleryItemStarredFromCaches(queryClient, items);
          const currentStoreFields = collectGalleryStoreKnownItemFields(queries.getSnapshot().projects, items);
          const safeCacheKeys = new Set(
            selectItemKeysUnchangedSince(requestedKeys, starred, (key) => currentCacheStarred.get(key))
          );
          const safeStoreKeys = new Set(
            selectItemKeysUnchangedSince(requestedKeys, starred, (key) => currentStoreFields.get(key)?.starred)
          );
          const restoreGroups = new Map<boolean, GalleryItemRef[]>();

          for (const item of items) {
            const priorStarred = previousStarred.get(toGalleryItemKey(item));

            if (priorStarred === undefined) {
              continue;
            }

            const group = restoreGroups.get(priorStarred) ?? [];

            group.push(item);
            restoreGroups.set(priorStarred, group);
          }

          for (const [priorStarred, refs] of restoreGroups) {
            patchStarredCacheOnly(
              refs.filter((ref) => safeCacheKeys.has(toGalleryItemKey(ref))),
              priorStarred
            );
            patchStarredStoreOnly(
              refs.filter((ref) => safeStoreKeys.has(toGalleryItemKey(ref))).map(toGalleryItemKey),
              priorStarred
            );
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

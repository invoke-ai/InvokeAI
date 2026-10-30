/**
 * The canvas's losses made undoable: discarding staged results, the results an accept sets aside, clearing the canvas
 * and restoring a point. Each reports what happened with an Undo, as the workbench does for reversible actions, and
 * every Undo captures its project id, never "the active project".
 */

import type { CanvasSnapshotContract, CanvasStagingCandidateContract } from '@workbench/canvas-engine/api';
import type { Notify } from '@workbench/useNotify';
import type { TFunction } from 'i18next';

import { createItemId } from '@platform/browser/randomUuid';
import { getCanvasEngine, isBlankCanvasDocument } from '@workbench/canvas-operations/api';
import {
  getCanvasStagingCandidateFingerprint,
  getCanvasStagingSlots,
  hasPendingCanvasResults,
} from '@workbench/canvasStagingView';
import { useNotify } from '@workbench/useNotify';
import { useActiveProjectId, useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import {
  clearCanvasWithRestorePoint,
  getInFlightCanvasItems,
  restoreFromPoint,
  saveRestorePoint,
  type RestoreOutcome,
  type RestorePointPorts,
} from './canvasRestorePoints';

/** One project's staged results and recovery: a newer staging Undo withdraws an older one's toast. */
const stagingUndoKey = (projectId: string): string => `canvas-staging:${projectId}`;
const restoreUndoKey = (projectId: string): string => `canvas-restore:${projectId}`;

const useRecoveryPorts = () => {
  const { canvas } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  return useCallback(
    (projectId: string): RestorePointPorts => ({
      apply: (mutation) => canvas.apply(projectId, mutation),
      createId: () => createItemId('restore-point'),
      // Looked up when the action runs: an Undo may come after the canvas that offered it closed.
      engine: () => getCanvasEngine(projectId) ?? null,
      isBlank: () => {
        const project = queries.getProject(projectId);
        return (
          !!project &&
          project.canvas.stagingArea.pendingImages.length === 0 &&
          isBlankCanvasDocument(project.canvas.document)
        );
      },
      lockReason: () => {
        const project = queries.getProject(projectId);
        if (getCanvasEngine(projectId)?.interaction.get('documentEditingLocked')) {
          return 'editing';
        }
        return project && hasPendingCanvasResults(project.canvas, project.queue.items) ? 'canvas' : null;
      },
      now: () => new Date().toISOString(),
    }),
    [canvas, queries]
  );
};

/** Says why a restore changed nothing; nothing when it did. */
const explainRestore = (outcome: RestoreOutcome, notify: Notify, t: TFunction): void => {
  switch (outcome.status) {
    case 'restored':
      return;
    case 'locked':
      notify.info(
        t('widgets.canvas.restorePoints.notRestored'),
        outcome.reason === 'editing'
          ? t('widgets.canvas.controls.operationRunning')
          : t('widgets.canvas.restorePoints.lockedByGeneration')
      );
      return;
    case 'unsaved':
      notify.error(t('widgets.canvas.restorePoints.notRestored'), t('widgets.canvas.restorePoints.unsavedMessage'));
      return;
    case 'nothing':
      notify.info(t('widgets.canvas.restorePoints.alreadyStaged'));
      return;
    case 'gone':
      notify.error(t('widgets.canvas.restorePoints.unavailable'), t('widgets.canvas.restorePoints.unavailableMessage'));
  }
};

/**
 * Undo of a Clear canvas or of a restore: brings the point back like the History pane does, keeping what the canvas
 * holds now as a point of its own, and says why when it cannot.
 */
const undoWithPoint = async (
  ports: RestorePointPorts,
  pointId: string,
  pointName: string,
  notify: Notify,
  t: TFunction
): Promise<void> => {
  const outcome = await restoreFromPoint(
    ports,
    { id: pointId, kind: 'clear-canvas' },
    t('widgets.canvas.restorePoints.beforeRestore', { name: pointName })
  );
  explainRestore(outcome, notify, t);
  if (outcome.status === 'restored' && outcome.undoPointId) {
    notify.success(
      t('widgets.canvas.restorePoints.restored', { name: pointName }),
      t('widgets.canvas.restorePoints.keptReplaced')
    );
  }
};

/** Discard, Discard all and accept, each reporting what left staging with a way to bring it back. */
export const useStagingRecovery = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const { canvas, queue } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const projectId = useActiveProjectId();
  const portsFor = useRecoveryPorts();

  const offerRestore = useCallback(
    (
      targetProjectId: string,
      title: string,
      notice: { message?: string; undoLabel?: string },
      images: readonly CanvasStagingCandidateContract[],
      previous: readonly CanvasStagingCandidateContract[],
      documentRevision: number,
      /** A restore point of the same results, used up once they are back. */
      pointId: string | null = null
    ) =>
      notify.undoable(title, {
        ...notice,
        onUndo: () => {
          if (canvas.apply(targetProjectId, { documentRevision, images, previous, type: 'restoreStagedImages' })) {
            if (pointId) {
              canvas.apply(targetProjectId, { snapshotId: pointId, type: 'deleteCanvasSnapshot' });
            }
            return;
          }
          // The canvas was replaced since (its results went with it); the point re-stages them on the new one.
          if (pointId) {
            void restoreFromPoint(portsFor(targetProjectId), { id: pointId, kind: 'discard-results' }, '').then(
              (outcome) => explainRestore(outcome, notify, t)
            );
            return;
          }
          notify.info(t('widgets.canvas.staging.notBroughtBack'), t('widgets.canvas.staging.notBroughtBackMessage'));
        },
        projectId: targetProjectId,
        supersedes: stagingUndoKey(targetProjectId),
      }),
    [canvas, notify, portsFor, t]
  );

  /** The selected candidate, read when the discard runs so a result that arrived meanwhile is never the one dropped. */
  const discardSelected = useCallback(() => {
    const project = queries.getProject(projectId);
    if (!project) {
      return;
    }
    const { documentRevision, stagingArea } = project.canvas;
    const slot = getCanvasStagingSlots(project.canvas, project.queue.items)[stagingArea.selectedImageIndex];
    if (slot?.kind !== 'candidate') {
      return;
    }
    if (canvas.apply(projectId, { type: 'discardSelectedStagedImage' })) {
      offerRestore(
        projectId,
        t('widgets.canvas.staging.discarded'),
        {},
        [slot.candidate],
        stagingArea.pendingImages,
        documentRevision
      );
    }
  }, [canvas, offerRestore, projectId, queries, t]);

  const discardAll = useCallback(() => {
    const project = queries.getProject(projectId);
    if (!project) {
      return;
    }
    const { documentRevision, stagingArea } = project.canvas;
    const images = stagingArea.pendingImages;
    const inFlight = getInFlightCanvasItems(project.canvas, project.queue.items);
    const pointId =
      images.length > 0
        ? saveRestorePoint(portsFor(projectId), 'discard-results', t('widgets.canvas.restorePoints.discardedResults'))
        : null;
    canvas.apply(projectId, { type: 'discardAllStagedImages' });
    // "All" includes what is still generating for this canvas; the notice below says so, not a cancel notice each.
    for (const item of inFlight) {
      queue.cancel(projectId, item.id, null);
    }
    if (images.length === 0 && inFlight.length === 0) {
      return;
    }
    if (images.length === 0) {
      notify.info(t('widgets.canvas.staging.stoppedGenerating', { count: inFlight.length }));
      return;
    }
    offerRestore(
      projectId,
      t('widgets.canvas.staging.discardedAll', { count: images.length }),
      {
        message:
          inFlight.length > 0
            ? t('widgets.canvas.staging.discardedAllStopped', { count: inFlight.length })
            : t('widgets.canvas.staging.discardedAllKept'),
      },
      images,
      images,
      documentRevision,
      pointId
    );
  }, [canvas, notify, offerRestore, portsFor, projectId, queries, queue, t]);

  /**
   * After an accept that set other results aside: says so (and that it stopped the batch) and offers them back for
   * review. The accepted layer stays; undoing it on the canvas removes it without re-staging it.
   */
  const reportAccepted = useCallback(
    (
      before: { documentRevision: number; pendingImages: readonly CanvasStagingCandidateContract[] },
      accepted: CanvasStagingCandidateContract,
      stoppedBatch: boolean
    ) => {
      const fingerprint = getCanvasStagingCandidateFingerprint(accepted);
      const others = before.pendingImages.filter(
        (image) => getCanvasStagingCandidateFingerprint(image) !== fingerprint
      );
      if (others.length === 0) {
        return;
      }
      const setAside = t('widgets.canvas.staging.acceptedOthers', { count: others.length });
      offerRestore(
        projectId,
        t('widgets.canvas.staging.acceptStopTitle'),
        {
          message: stoppedBatch ? `${setAside} ${t('widgets.canvas.staging.acceptedStopped')}` : setAside,
          undoLabel: t('widgets.canvas.staging.reviewOthers', { count: others.length }),
        },
        others,
        before.pendingImages,
        before.documentRevision
      );
    },
    [offerRestore, projectId, t]
  );

  return useMemo(
    () => ({ discardAll, discardSelected, reportAccepted }),
    [discardAll, discardSelected, reportAccepted]
  );
};

/** Clears the canvas behind a restore point and offers Undo; nothing is cleared when the latest strokes cannot be saved. */
export const useClearCanvas = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const { queue } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const projectId = useActiveProjectId();
  const portsFor = useRecoveryPorts();

  return useCallback(
    async (stopGenerations: boolean): Promise<void> => {
      const ports = portsFor(projectId);
      const name = t('widgets.canvas.restorePoints.beforeClear');
      const outcome = await clearCanvasWithRestorePoint({
        cancelQueueItem: (queueItemId) => queue.cancel(projectId, queueItemId, null),
        name,
        ports,
        readTarget: () => {
          const project = queries.getProject(projectId);
          return project
            ? {
                document: project.canvas.document,
                stop: stopGenerations
                  ? getInFlightCanvasItems(project.canvas, project.queue.items).map((item) => item.id)
                  : [],
              }
            : null;
        },
      });
      switch (outcome.status) {
        case 'gone':
          return;
        case 'unsaved':
          notify.error(t('widgets.canvas.controls.clearUnsaved'), t('widgets.canvas.controls.clearUnsavedMessage'));
          return;
        case 'failed':
          notify.error(t('widgets.canvas.controls.clearUnsaved'), t('widgets.canvas.controls.clearFailedMessage'));
          return;
      }
      const { restorePointId, stopped: stoppedCount } = outcome;
      const stopped = stoppedCount > 0 ? t('widgets.canvas.controls.clearedStopped', { count: stoppedCount }) : null;
      if (restorePointId === null) {
        notify.success(t('widgets.canvas.controls.cleared'), stopped ?? undefined);
        return;
      }
      notify.undoable(t('widgets.canvas.controls.cleared'), {
        message: [t('widgets.canvas.controls.clearedMessage'), stopped].filter(Boolean).join(' '),
        onUndo: () => void undoWithPoint(ports, restorePointId, name, notify, t),
        projectId,
        supersedes: restoreUndoKey(projectId),
      });
    },
    [notify, portsFor, projectId, queries, queue, t]
  );
};

/** Restores a point from the History pane; a canvas restore keeps the replaced canvas as the point its Undo restores. */
export const useRestorePoint = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const projectId = useActiveProjectId();
  const portsFor = useRecoveryPorts();

  return useCallback(
    async (point: Pick<CanvasSnapshotContract, 'id' | 'kind' | 'name'>): Promise<void> => {
      const ports = portsFor(projectId);
      const beforeName = t('widgets.canvas.restorePoints.beforeRestore', { name: point.name });
      const outcome = await restoreFromPoint(ports, point, beforeName);
      if (outcome.status !== 'restored') {
        explainRestore(outcome, notify, t);
        return;
      }
      const { undoPointId } = outcome;
      if (point.kind === 'discard-results') {
        notify.success(t('widgets.canvas.restorePoints.resultsBack'));
        return;
      }
      if (undoPointId === null) {
        notify.success(t('widgets.canvas.restorePoints.restored', { name: point.name }));
        return;
      }
      notify.undoable(t('widgets.canvas.restorePoints.restored', { name: point.name }), {
        onUndo: () => void undoWithPoint(ports, undoPointId, beforeName, notify, t),
        projectId,
        supersedes: restoreUndoKey(projectId),
      });
    },
    [notify, portsFor, projectId, t]
  );
};

import type { ModelConfig } from '@features/models';
import type { QueueGenerationMeta } from '@features/queue/contracts';
import type { ImageRecallKind } from '@workbench/image-actions';
import type * as CanvasRecallModule from '@workbench/image-actions/canvasRecall';
import type { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { isSupportedGenerateModel } from '@features/generation/settings';
import { hasArchitectureCapabilities } from '@features/models/architectures';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { claimCanvasRecall, getCurrentGenerateValues, getImageRecallTitle } from '@workbench/image-actions';
import { getResolvedProjectWidgetValues } from '@workbench/widgetState';

import type { LocalRecallSnapshot } from './useLocalRecallSnapshot';

import {
  getQueueRecallAvailability,
  getQueueRecallSource,
  getQueueRecallTargetState,
  planQueueRecall,
} from './queueRecall';

/** What a queue item recall writes through, and how it reports: the Workbench's commands, read side and notices. */
export interface QueueItemRecallContext {
  commands: WorkbenchCommands;
  notify: { info(title: string, description?: string): void; success(title: string, description?: string): void };
  openWidget: ReturnType<typeof useOpenWorkbenchWidget>;
  queries: WorkbenchQueries;
  t: TFunction;
}

/**
 * Recalls a local submission snapshot, or a foreign item's session prompts and seed, into the active project, then
 * reveals the destination panel. An unavailable kind writes nothing and says why. A Canvas recall pressed while
 * another runs says so (see `claimCanvasRecall`).
 */
export const executeQueueItemRecall = async ({
  context,
  isNonGenerateRun,
  kind,
  meta,
  models,
  snapshot,
}: {
  context: QueueItemRecallContext;
  /** The server names a workflow the item ran; a snapshot, when this browser kept one, names its source instead. */
  isNonGenerateRun: boolean;
  kind: ImageRecallKind;
  meta: QueueGenerationMeta;
  models: readonly ModelConfig[];
  /** Null when this browser kept no snapshot for the item. */
  snapshot: LocalRecallSnapshot | null;
}): Promise<void> => {
  const { commands, notify, openWidget, queries, t } = context;
  const { generation, widgets } = commands;
  const owner = captureAccountScope();
  const originalProject = queries.getSnapshot().activeProject;
  const supportedModels = models.filter(isSupportedGenerateModel);
  const localGenerateValues = snapshot?.generateValues ?? null;
  const videoSnapshot = snapshot?.videoValues ?? null;
  const isVideoItem = snapshot?.sourceId === 'video';
  const readCurrentValues = () => {
    const project = queries.getProject(originalProject.id);
    return project
      ? getCurrentGenerateValues({
          generateValues: getResolvedProjectWidgetValues(project, 'generate', queries.getSnapshot().scopedDefaults),
          supportedModels,
        })
      : null;
  };
  const current = readCurrentValues();
  const { reasons } = getQueueRecallAvailability({
    meta,
    snapshot: localGenerateValues,
    source: getQueueRecallSource(snapshot?.sourceId ?? null, isNonGenerateRun),
    // Callers read the catalog before recalling; one that failed to load leaves no model to select.
    target: getQueueRecallTargetState({
      hasCapabilityTable: hasArchitectureCapabilities(),
      hasCurrentValues: current !== null,
      isCatalogSettled: true,
    }),
    videoSnapshot,
  });
  const unavailableReason = reasons[kind];
  let plan = unavailableReason
    ? null
    : planQueueRecall(kind, { current, isVideoItem, meta, snapshot: localGenerateValues, videoSnapshot });

  if (!plan) {
    notify.info(t('widgets.queue.nothingRecalled'), t(unavailableReason ?? 'widgets.queue.recallUnavailableForItem'));
    return;
  }

  if (plan.target === 'video') {
    widgets.patchValues('video', plan.patch);
    openWidget('video', { preferredRegions: ['left'] });
    notify.success(getImageRecallTitle(kind, t), t('widgets.queue.settingsRecalledIntoVideoDescription'));
    return;
  }

  let canvasRecall: typeof CanvasRecallModule | null = null;
  let canvasPlan: CanvasRecallModule.CanvasRecallPlan | null = null;
  let canvasProblem: string | null = null;
  let canvasApplied = false;
  const preparedModelKey = plan.values.model?.key;
  // Released once committed: saving the Canvas images afterwards holds no later recall back.
  let releaseCanvasRecall: (() => void) | null = null;
  try {
    if ((kind === 'all' || kind === 'remix') && snapshot?.canvasMetadata) {
      releaseCanvasRecall = claimCanvasRecall(owner);
      if (!releaseCanvasRecall) {
        notify.info(getImageRecallTitle(kind, t), t('widgets.gallery.itemActions.recall.recallInProgress'));
        return;
      }
      try {
        canvasRecall = await import('@workbench/image-actions/canvasRecall');
        canvasPlan = await canvasRecall.prepareCanvasRecall({
          metadata: snapshot.canvasMetadata,
          model: plan.values.model,
          models: [...models],
          owner,
          isCurrent: () =>
            queries.isActiveProject(originalProject.id) &&
            queries.getProject(originalProject.id)?.canvas.document === originalProject.canvas.document,
        });
        await canvasRecall.settleCanvasRecall(
          canvasPlan,
          {
            commands,
            projectId: originalProject.id,
            getProject: () => queries.getProject(originalProject.id) ?? null,
          },
          owner
        );
      } catch (error) {
        canvasPlan = null;
        canvasProblem = error instanceof Error ? error.message : 'unavailable';
      }
    }
    if (!isAccountScopeCurrent(owner) || !queries.isActiveProject(originalProject.id)) {
      return;
    }
    // Asset reads and paint persistence may yield; retain edits made while the canvas was loading.
    plan = planQueueRecall(kind, {
      current: readCurrentValues(),
      isVideoItem,
      meta,
      snapshot: localGenerateValues,
      videoSnapshot,
    });
    if (!plan || plan.target !== 'generate') {
      return;
    }
    if (canvasPlan && canvasRecall) {
      try {
        if (preparedModelKey !== plan.values.model?.key) {
          canvasPlan.problems.push('model-changed');
        } else {
          canvasApplied = canvasRecall.commitCanvasRecall({
            plan: canvasPlan,
            expectedDocument: originalProject.canvas.document,
            target: {
              commands,
              projectId: originalProject.id,
              getProject: () =>
                queries.isActiveProject(originalProject.id) ? (queries.getProject(originalProject.id) ?? null) : null,
            },
            restorePointName: t('widgets.gallery.itemActions.recall.beforeCanvasRecall'),
          });
        }
        if (canvasPlan.problems.length) {
          canvasProblem = canvasRecall.getCanvasRecallProblemsMessage(canvasPlan.problems, t);
        }
      } catch (error) {
        canvasProblem = error instanceof Error ? error.message : 'unavailable';
      }
    }
    if (canvasProblem) {
      notify.info(
        t('widgets.gallery.itemActions.recall.partiallyRecalled'),
        t('widgets.gallery.itemActions.recall.canvasPartial', { items: canvasProblem })
      );
    }
    generation.setSettings(plan.values, originalProject.id);
    openWidget('generate', { preferredRegions: ['left'] });
    notify.success(getImageRecallTitle(kind, t), t('widgets.queue.settingsRecalledDescription'));
  } finally {
    releaseCanvasRecall?.();
  }
  if (canvasApplied && canvasPlan && canvasRecall) {
    const durable = await canvasRecall.finalizeCanvasRecallAssets(
      canvasPlan,
      {
        commands,
        projectId: originalProject.id,
        getProject: () => queries.getProject(originalProject.id) ?? null,
      },
      owner
    );
    if (!durable && isAccountScopeCurrent(owner) && queries.isActiveProject(originalProject.id)) {
      notify.info(getImageRecallTitle(kind, t), t('widgets.gallery.itemActions.recall.canvasImagesTemporary'));
    }
  }
};

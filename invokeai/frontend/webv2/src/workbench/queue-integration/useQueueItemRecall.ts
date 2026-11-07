import type { QueueGenerationMeta } from '@features/queue/contracts';
import type { ImageRecallCapabilities, ImageRecallKind } from '@workbench/image-actions';
import type * as CanvasRecallModule from '@workbench/image-actions/canvasRecall';

import { hasArchitectureCapabilities, isSupportedGenerateModel } from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import {
  EMPTY_IMAGE_RECALL_CAPABILITIES,
  getCurrentGenerateValues,
  getImageRecallTitle,
} from '@workbench/image-actions';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { getQueueRecallCapabilities, getVideoQueueRecallCapabilities, planQueueRecall } from './queueRecall';
import { useLocalRecallSnapshot } from './useLocalRecallSnapshot';

/**
 * Recall local submission snapshots or foreign session prompts/seed into the active project, then reveal the
 * destination panel.
 */
export const useQueueItemRecall = (
  origin: string | null | undefined,
  meta: QueueGenerationMeta
): { capabilities: ImageRecallCapabilities; isPending: boolean; recall: (kind: ImageRecallKind) => void } => {
  const { t } = useTranslation();
  const commands = useWorkbenchCommands();
  const { generation, widgets } = commands;
  const queries = useWorkbenchQueries();
  const openWidget = useOpenWorkbenchWidget();
  const notify = useNotify();
  const snapshot = useLocalRecallSnapshot(origin);
  const isPending = snapshot === undefined;
  const localGenerateValues = snapshot?.generateValues ?? null;
  const videoSnapshot = snapshot?.videoValues ?? null;
  const isVideoItem = snapshot?.sourceId === 'video';
  const models = useModelsSelector((state) => state.models);
  const supportedModels = useMemo(() => models.filter(isSupportedGenerateModel), [models]);
  const capabilities = useMemo(
    () =>
      isPending
        ? EMPTY_IMAGE_RECALL_CAPABILITIES
        : isVideoItem
          ? getVideoQueueRecallCapabilities(videoSnapshot, meta)
          : getQueueRecallCapabilities(localGenerateValues, meta),
    [isPending, isVideoItem, localGenerateValues, meta, videoSnapshot]
  );

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const recall = useCallback(
    async (kind: ImageRecallKind) => {
      const owner = captureAccountScope();
      const originalProject = queries.getSnapshot().activeProject;
      if (isPending) {
        return;
      }
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
      let plan = planQueueRecall(kind, { current, isVideoItem, meta, snapshot: localGenerateValues, videoSnapshot });

      if (!plan) {
        // Use a generic unavailable reason: Video and missing capabilities are not missing Generate models.
        const reason = isVideoItem
          ? 'widgets.queue.recallUnavailableForItem'
          : !current && !hasArchitectureCapabilities()
            ? 'widgets.queue.recallUnavailableCapabilities'
            : 'widgets.queue.recallUnavailable';
        notify.info(getImageRecallTitle(kind, t), t(reason));
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
      if ((kind === 'all' || kind === 'remix') && snapshot?.canvasMetadata) {
        try {
          canvasRecall = await import('@workbench/image-actions/canvasRecall');
          canvasPlan = await canvasRecall.prepareCanvasRecall({
            metadata: snapshot.canvasMetadata,
            model: plan.values.model,
            models,
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
          getImageRecallTitle(kind, t),
          t('widgets.gallery.itemActions.recall.canvasPartial', { items: canvasProblem })
        );
      }
      generation.setSettings(plan.values, originalProject.id);
      openWidget('generate', { preferredRegions: ['left'] });
      notify.success(getImageRecallTitle(kind, t), t('widgets.queue.settingsRecalledDescription'));
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
    },
    [
      commands,
      queries,
      snapshot,
      models,
      generation,
      isPending,
      isVideoItem,
      localGenerateValues,
      meta,
      notify,
      openWidget,
      supportedModels,
      t,
      videoSnapshot,
      widgets,
    ]
  );

  return { capabilities, isPending, recall };
};

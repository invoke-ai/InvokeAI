import type { QueueGenerationMeta } from '@features/queue/contracts';
import type { ImageRecallKind } from '@workbench/image-actions';

import { createGenerateFormValuesSelector } from '@features/generation/react';
import { getArchitectureCapabilitiesSnapshot, subscribeArchitectureCapabilities } from '@features/generation/runtime';
import { hasArchitectureCapabilities, isSupportedGenerateModel } from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { EMPTY_IMAGE_RECALL_CAPABILITIES, getCurrentGenerateValues } from '@workbench/image-actions';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useWidgetValuesSelector, useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { executeQueueItemRecall } from './executeQueueItemRecall';
import {
  getQueueRecallAvailability,
  getQueueRecallSource,
  getQueueRecallTargetState,
  type QueueRecallAvailability,
} from './queueRecall';
import { useLocalRecallSnapshot } from './useLocalRecallSnapshot';

export interface QueueItemRecall extends QueueRecallAvailability {
  /** The durable snapshot lookup has not answered yet: everything is unavailable, with no reason to give. */
  isPending: boolean;
  recall: (kind: ImageRecallKind) => Promise<void>;
}

const selectGenerateValues = createGenerateFormValuesSelector();
const NO_REASONS: QueueRecallAvailability['reasons'] = {};

/**
 * What an item's recall menus offer, and the recall itself. Mount it only for the item whose menu or details are
 * open: it reads the durable snapshot and loads the model catalog.
 */
export const useQueueItemRecall = (
  origin: string | null | undefined,
  meta: QueueGenerationMeta,
  /** The server names a workflow the item ran: without a snapshot, the only sign it holds no Generate settings. */
  { isNonGenerateRun = false }: { isNonGenerateRun?: boolean } = {}
): QueueItemRecall => {
  const { t } = useTranslation();
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const openWidget = useOpenWorkbenchWidget();
  const notify = useNotify();
  const snapshot = useLocalRecallSnapshot(origin);
  const isPending = snapshot === undefined;
  const models = useModelsSelector((state) => state.models);
  const isCatalogSettled = useModelsSelector((state) => state.status === 'loaded' || state.status === 'error');
  const supportedModels = useMemo(() => models.filter(isSupportedGenerateModel), [models]);
  const generateValues = useWidgetValuesSelector('generate', selectGenerateValues);
  // Read under the capability store: Generate values synthesized from architecture policy appear with its table.
  const target = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(
      () =>
        getQueueRecallTargetState({
          hasCapabilityTable: hasArchitectureCapabilities(),
          hasCurrentValues: getCurrentGenerateValues({ generateValues, models, supportedModels }) !== null,
          isCatalogSettled,
        }),
      [generateValues, isCatalogSettled, models, supportedModels]
    )
  );
  const source = getQueueRecallSource(snapshot?.sourceId ?? null, isNonGenerateRun);
  const generateSnapshot = snapshot?.generateValues ?? null;
  const videoSnapshot = snapshot?.videoValues ?? null;
  const availability = useMemo<QueueRecallAvailability>(
    () =>
      isPending
        ? { capabilities: EMPTY_IMAGE_RECALL_CAPABILITIES, reasons: NO_REASONS }
        : getQueueRecallAvailability({ meta, snapshot: generateSnapshot, source, target, videoSnapshot }),
    [generateSnapshot, isPending, meta, source, target, videoSnapshot]
  );

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const recall = useCallback(
    (kind: ImageRecallKind) => {
      if (snapshot === undefined) {
        return Promise.resolve();
      }
      return executeQueueItemRecall({
        context: { commands, notify, openWidget, queries, t },
        isNonGenerateRun,
        kind,
        meta,
        models,
        snapshot,
      });
    },
    [commands, isNonGenerateRun, meta, models, notify, openWidget, queries, snapshot, t]
  );

  return { ...availability, isPending, recall };
};

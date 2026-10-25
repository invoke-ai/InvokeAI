import type {
  CanvasDocumentCapability,
  CanvasExportCapability,
  CanvasToolCapability,
} from '@workbench/canvas-engine/api';
import type {
  CanvasCompositeExportEngine,
  ControlLayerIntent,
  ControlLayerIntentResult,
  ControlSourceNotice,
  ControlSourceSnapshot,
} from '@workbench/canvas-operations/api';

import { invalidateGallery } from '@features/gallery/queries';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { useQueryClient } from '@tanstack/react-query';
import {
  fillControlFromBbox,
  getControlIntentNotice,
  getControlSourceSnapshot,
  getControlUploadNotice,
  subscribeControlSourceSnapshot,
  uploadControlImageFile,
} from '@workbench/canvas-operations/api';
import { useNotify } from '@workbench/useNotify';
import { describeAdjustmentRefusal } from '@workbench/widgets/canvas/useStructuralCommit';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

export type ControlSourceAction = 'upload' | 'frame' | 'draw';

/** The engine surface the Image row's actions need. */
export type ControlSourceEngine = CanvasCompositeExportEngine & {
  readonly document: CanvasDocumentCapability;
  readonly exports: Pick<CanvasExportCapability, 'hasExportableLayerContent'>;
  readonly projectId: string;
  readonly tools: Pick<CanvasToolCapability, 'setTool'>;
};

/** Stands in for a missing engine: it has no source state, so it reads the empty snapshot. */
const NO_ENGINE = {};
const noSubscription = (): (() => void) => () => undefined;
const selectSnapshot = (snapshot: ControlSourceSnapshot) => snapshot;

/** Shows a control source notice through the workbench notices. */
const useControlNoticeReporter = (): ((notice: ControlSourceNotice | null) => void) => {
  const { t } = useTranslation();
  const notify = useNotify();
  return useCallback(
    (notice: ControlSourceNotice | null) => {
      if (notice) {
        notify[notice.kind](t(notice.titleKey), notice.messageKey ? t(notice.messageKey) : notice.message);
      }
    },
    [notify, t]
  );
};

/** Says why a control layer intent did not land; `updated`, `unchanged` and `missing` say nothing. */
export const useControlIntentReporter = (): ((result: ControlLayerIntentResult) => void) => {
  const report = useControlNoticeReporter();
  return useCallback((result: ControlLayerIntentResult) => report(getControlIntentNotice(result)), [report]);
};

/**
 * The Image row's three ways to fill a control layer. Each ends in one Properties intent (`replace-image` or `draw`),
 * so the coordinator detects the map, and reports a refusal. An action outlives the panel: its result still lands on
 * the layer by id. An upload started from the canvas shows here as busy too.
 */
export const useControlSourceActions = ({
  applyIntent,
  engine,
  layerId,
}: {
  /** Applies an intent and reports a refusal. */
  applyIntent: (intent: ControlLayerIntent) => ControlLayerIntentResult | null;
  engine: ControlSourceEngine | null;
  layerId: string;
}) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const queryClient = useQueryClient();
  const report = useControlNoticeReporter();
  const reportIntent = useControlIntentReporter();
  const [isFillingFromFrame, setIsFillingFromFrame] = useState(false);
  const subscribe = useCallback(
    (listener: () => void) => (engine ? subscribeControlSourceSnapshot(engine, listener) : noSubscription()),
    [engine]
  );
  const getSnapshot = useCallback(() => getControlSourceSnapshot(engine ?? NO_ENGINE), [engine]);
  const { origins, uploading } = useExternalStoreSelector(subscribe, getSnapshot, selectSnapshot);
  const busy: ControlSourceAction | null = isFillingFromFrame ? 'frame' : uploading.has(layerId) ? 'upload' : null;

  const upload = useCallback(
    async (file: File) => {
      if (!engine || isFillingFromFrame) {
        return;
      }
      const outcome = await uploadControlImageFile({
        engine,
        file,
        layerId,
        onUploaded: (owner) => void invalidateGallery(queryClient, owner),
      });
      report(getControlUploadNotice(outcome));
    },
    [engine, isFillingFromFrame, layerId, queryClient, report]
  );

  const fromFrame = useCallback(async () => {
    if (!engine || busy) {
      return;
    }
    setIsFillingFromFrame(true);
    try {
      const filled = await fillControlFromBbox({ engine, layerId });
      switch (filled.status) {
        case 'applied':
          reportIntent(filled.result);
          return;
        case 'empty':
          notify.info(t('widgets.layers.control.source.frameEmpty'));
          return;
        case 'not-ready':
        case 'over-budget':
          notify.info(t('widgets.layers.control.source.frameNotReady'));
          return;
        case 'adjustment-refused':
          notify.error(t('widgets.layers.control.source.frameFailed'), describeAdjustmentRefusal(filled, t));
          return;
        case 'stale':
        // The account changed on the way; the result belongs to nobody here.
      }
    } catch (error) {
      notify.error(
        t('widgets.layers.control.source.frameFailed'),
        error instanceof Error ? error.message : String(error)
      );
    } finally {
      setIsFillingFromFrame(false);
    }
  }, [busy, engine, layerId, notify, reportIntent, t]);

  const draw = useCallback(() => {
    const result = applyIntent({ kind: 'draw' });
    if (result === 'updated' || result === 'unchanged') {
      engine?.tools.setTool('brush');
    }
  }, [applyIntent, engine]);

  return { busy, draw, fromFrame, origins, upload };
};

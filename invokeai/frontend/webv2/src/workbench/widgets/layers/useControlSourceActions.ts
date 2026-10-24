import type { CanvasDocumentCapability, CanvasToolCapability } from '@workbench/canvas-engine/api';
import type {
  CanvasCompositeExportEngine,
  ControlLayerIntent,
  ControlLayerIntentResult,
} from '@workbench/canvas-operations/api';

import { galleryTransfers } from '@features/gallery';
import { invalidateGallery } from '@features/gallery/queries';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { useQueryClient } from '@tanstack/react-query';
import { fillControlFromBbox } from '@workbench/canvas-operations/api';
import { useNotify } from '@workbench/useNotify';
import { describeAdjustmentRefusal } from '@workbench/widgets/canvas/useStructuralCommit';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

export type ControlSourceAction = 'upload' | 'frame' | 'draw';

/** The engine surface the Image row's actions need. */
export type ControlSourceEngine = CanvasCompositeExportEngine & {
  readonly document: CanvasDocumentCapability;
  readonly projectId: string;
  readonly tools: Pick<CanvasToolCapability, 'setTool'>;
};

/** Where an image this panel placed came from; images placed elsewhere are described by their size. */
export type ControlSourceOrigin = { kind: 'file'; name: string } | { kind: 'frame' };

/**
 * The Image row's three ways to fill a control layer. Each ends in one Properties intent (`replace-image` or `draw`),
 * so the coordinator detects the map; `applyIntent` reports refusals. An action outlives the panel: its result still
 * lands on the layer by id.
 */
export const useControlSourceActions = ({
  applyIntent,
  engine,
  layerId,
  reportIntent,
}: {
  /** Applies an intent and reports a refusal. */
  applyIntent: (intent: ControlLayerIntent) => ControlLayerIntentResult | null;
  /** Reports a refusal of an intent applied elsewhere. */
  reportIntent: (result: ControlLayerIntentResult) => void;
  engine: ControlSourceEngine | null;
  layerId: string;
}) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const queryClient = useQueryClient();
  const [busy, setBusy] = useState<ControlSourceAction | null>(null);
  const [origins, setOrigins] = useState<Readonly<Record<string, ControlSourceOrigin>>>({});
  const remember = useCallback(
    (imageName: string, origin: ControlSourceOrigin) => setOrigins((current) => ({ ...current, [imageName]: origin })),
    []
  );

  const upload = useCallback(
    async (file: File) => {
      if (!engine || busy) {
        return;
      }
      const owner = captureAccountScope();
      setBusy('upload');
      try {
        const uploaded = await galleryTransfers.upload(file, 'none', { signal: owner.signal });
        assertAccountScopeCurrent(owner);
        const bbox = engine.document.getDocument()?.bbox;
        const result = applyIntent({
          image: { height: uploaded.height, imageName: uploaded.imageName, width: uploaded.width },
          kind: 'replace-image',
          // New images land at the bbox origin, as every other canvas import does.
          transform: bbox ? { rotation: 0, scaleX: 1, scaleY: 1, x: bbox.x, y: bbox.y } : undefined,
        });
        if (result === 'updated') {
          remember(uploaded.imageName, { kind: 'file', name: file.name });
        }
        void invalidateGallery(queryClient, owner);
      } catch (error) {
        if (isAccountScopeCurrent(owner)) {
          notify.error(
            t('widgets.layers.control.source.uploadFailed'),
            error instanceof Error ? error.message : String(error)
          );
        }
      } finally {
        setBusy(null);
      }
    },
    [applyIntent, busy, engine, notify, queryClient, remember, t]
  );

  const fromFrame = useCallback(async () => {
    if (!engine || busy) {
      return;
    }
    setBusy('frame');
    try {
      const filled = await fillControlFromBbox({ engine, layerId });
      switch (filled.status) {
        case 'applied':
          if (filled.result === 'updated') {
            remember(filled.imageName, { kind: 'frame' });
          }
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
      setBusy(null);
    }
  }, [busy, engine, layerId, notify, remember, reportIntent, t]);

  const draw = useCallback(() => {
    const result = applyIntent({ kind: 'draw' });
    if (result === 'updated' || result === 'unchanged') {
      engine?.tools.setTool('brush');
    }
  }, [applyIntent, engine]);

  return { busy, draw, fromFrame, origins, upload };
};

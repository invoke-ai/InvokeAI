import type { ColorSamplePhase } from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { useMountEffect } from '@platform/react/useMountEffect';
import { usePreparedCommit, useStructuralPreview } from '@workbench/widgets/canvas/useStructuralCommit';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useEffect, useEffectEvent, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { ActiveColorTarget } from './colorPair';

import { readActiveColorPair, readActiveColorTarget } from './colorPair';
import { maskTintTargetStore, settlePendingMaskTint } from './maskTintTarget';
import { useActiveColorCommands, useActiveColorPair } from './useActiveColors';
import {
  armedMaskLayer,
  isMaskLayer,
  MASK_TINT_EDIT_LABEL_KEY,
  maskTintPreview,
  prepareMaskTintEdit,
} from './useMaskTintEditor';

/**
 * What one eyedropper pick writes, fixed at its first sample so a selection change mid-pick cannot retarget it.
 * Gradient stops have no armed state, so a pick on the Gradient tool writes the pair. A live text session is no
 * target: its editor holds focus and keeps Alt, and leaving it commits the session.
 */
type PickTarget =
  | { kind: 'mask-tint'; layerId: string }
  | { kind: 'shape'; layerId: string; field: 'fill' | 'stroke' }
  | { kind: 'text-layer'; layerId: string }
  | { kind: 'pair'; target: ActiveColorTarget; before: string };

/**
 * Precedence: the armed mask tint while its mask is still selected; a selected shape (the foreground target writes
 * its fill, the background its stroke); a selected text layer; otherwise the pair.
 */
const resolvePickTarget = (engine: CanvasEngineHandle, canvasValues: Record<string, unknown>): PickTarget => {
  const model = engine.document.model();
  const document = model?.document ?? null;
  const pairTarget = readActiveColorTarget(canvasValues);
  if (document) {
    const tint = armedMaskLayer(document, maskTintTargetStore.getSnapshot().layerId);
    if (tint) {
      return { kind: 'mask-tint', layerId: tint.id };
    }
  }
  const selected = document?.selectedLayerId ? (model?.getLayer(document.selectedLayerId) ?? null) : null;
  if (selected?.type === 'raster' && selected.source.type === 'shape') {
    return { field: pairTarget === 'foreground' ? 'fill' : 'stroke', kind: 'shape', layerId: selected.id };
  }
  if (selected?.type === 'raster' && selected.source.type === 'text') {
    return { kind: 'text-layer', layerId: selected.id };
  }
  return { before: readActiveColorPair(canvasValues)[pairTarget], kind: 'pair', target: pairTarget };
};

/**
 * Isolate frequent color-pair updates in this null adapter. Mirror the pair into brush color, captured at stroke
 * start, and route eyedropper picks to the armed color target.
 */
export const CanvasColorFeed = ({ engine }: { engine: CanvasEngineHandle | null }) => {
  const colorPair = useActiveColorPair();

  useEffect(() => {
    if (!engine) {
      return;
    }
    const options = engine.interaction.get('brushOptions');
    if (options.color !== colorPair.foreground) {
      engine.interaction.set('brushOptions', { ...options, color: colorPair.foreground });
    }
  }, [engine, colorPair.foreground]);

  // The whole pair, for gesture-start reads: new shapes and text sessions
  // capture it when they begin, and the gradient FG→BG preset resolves it.
  useEffect(() => {
    if (!engine) {
      return;
    }
    engine.interaction.set('colorPair', { background: colorPair.background, foreground: colorPair.foreground });
  }, [engine, colorPair.background, colorPair.foreground]);

  // Keyed by engine identity, so each engine instance (a project's engine recreated after disposal included) mounts
  // its own routing and registers its router exactly once.
  return engine ? <ColorSampleRouting key={engineKey(engine)} engine={engine} /> : null;
};

// Created on first use: a module-scope global constructor call would make this module side-effectful.
let engineKeys: WeakMap<CanvasEngineHandle, string> | null = null;
let nextEngineKey = 0;
/** A stable React key per engine object; held weakly, so a disposed engine's entry goes with it. */
const engineKey = (engine: CanvasEngineHandle): string => {
  engineKeys ??= new WeakMap();
  let key = engineKeys.get(engine);
  if (key === undefined) {
    nextEngineKey += 1;
    key = `engine-${nextEngineKey}`;
    engineKeys.set(engine, key);
  }
  return key;
};

/** Owns the engine's eyedropper router and the pick in progress. */
const ColorSampleRouting = ({ engine }: { engine: CanvasEngineHandle }) => {
  const { t } = useTranslation();
  const colorCommands = useActiveColorCommands();
  const queries = useWorkbenchQueries();
  const tintPreview = useStructuralPreview(engine);
  const commitPrepared = usePreparedCommit(engine);
  const pickRef = useRef<PickTarget | null>(null);

  /** Commits the shape's or text layer's color through the same edit and history label as its own picker. */
  const commitSourceColor = (target: Extract<PickTarget, { kind: 'shape' | 'text-layer' }>, hex: string): void => {
    const label =
      target.kind === 'shape' ? t('widgets.canvas.toolOptions.shapeEdit') : t('widgets.canvas.toolOptions.textEdit');
    commitPrepared(label, (model) => {
      const layer = model.getLayer(target.layerId);
      if (layer?.type !== 'raster') {
        return { ids: [target.layerId], status: 'missing' };
      }
      if (target.kind === 'shape' && layer.source.type === 'shape') {
        return model.prepare({ id: layer.id, source: { ...layer.source, [target.field]: hex }, type: 'patch-source' });
      }
      if (target.kind === 'text-layer' && layer.source.type === 'text') {
        return model.prepare({ id: layer.id, source: { ...layer.source, color: hex }, type: 'patch-source' });
      }
      return { ids: [target.layerId], status: 'missing' };
    });
  };

  // Previews are live and unrecorded; the release records one edit, and a cancel restores the color from before the
  // pick. Shape and text-layer colors live in the layer source, which has no live preview, so they change on release.
  const routeSample = useEffectEvent((hex: string, phase: ColorSamplePhase): boolean => {
    const startsPick = pickRef.current === null;
    const target =
      pickRef.current ??
      resolvePickTarget(engine, getProjectWidgetValues(queries.getSnapshot().activeProject, 'canvas'));
    pickRef.current = phase === 'preview' ? target : null;
    switch (target.kind) {
      case 'mask-tint': {
        if (phase === 'cancel') {
          tintPreview.cancel();
          break;
        }
        if (phase === 'preview') {
          // The pane's own tint gesture records first: the canvas holds one preview at a time.
          if (startsPick) {
            settlePendingMaskTint();
          }
          // The held layer, even if the selection moved mid-pick.
          const layer = engine.document.model()?.getLayer(target.layerId) ?? null;
          if (isMaskLayer(layer)) {
            tintPreview.preview(maskTintPreview(layer, hex));
          }
          break;
        }
        tintPreview.commit(t(MASK_TINT_EDIT_LABEL_KEY), (model, baseline) =>
          prepareMaskTintEdit(model, baseline, target.layerId, hex)
        );
        break;
      }
      case 'shape':
      case 'text-layer':
        if (phase === 'commit') {
          commitSourceColor(target, hex);
        }
        break;
      case 'pair':
        colorCommands.setPairColor(target.target, phase === 'cancel' ? target.before : hex);
        break;
    }
    return true;
  });

  /** A pick cut short by unmount records nothing. */
  const abandonPick = useEffectEvent((): void => {
    if (pickRef.current?.kind === 'mask-tint') {
      tintPreview.cancel();
    }
    pickRef.current = null;
  });

  // The parent keys this component by engine, so the router registers once per engine instance.
  // The linter cannot see useMountEffect as an effect, so it rejects calling the effect events from it.
  /* eslint-disable react-hooks/rules-of-hooks -- useMountEffect is the repository's explicit useEffect wrapper */
  useMountEffect(() => {
    const dispose = engine.tools.setColorSampleRouter((hex, phase) => routeSample(hex, phase));
    return () => {
      dispose();
      abandonPick();
    };
  });
  /* eslint-enable react-hooks/rules-of-hooks */

  return null;
};

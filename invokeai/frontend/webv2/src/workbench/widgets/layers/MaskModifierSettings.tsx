import type { CanvasInpaintMaskLayerContract } from '@workbench/canvas-engine/api';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';

import { ScrubberField } from '@platform/ui/ScrubberField';
import { type CanvasPreparedEngine, useStructuralPreview } from '@workbench/widgets/canvas/useStructuralCommit';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { MASK_MODIFIER_DEFAULTS } from './layerOps';

/** Edit modifier magnitude here with one commit per gesture; tree rows own enabling and removal. */

type MaskModifierKind = 'mask-noise' | 'mask-denoise';
type MaskModifier = NonNullable<CanvasInpaintMaskLayerContract['noise' | 'denoise']>;

const FIELD_OF: Record<MaskModifierKind, 'noise' | 'denoise'> = { 'mask-denoise': 'denoise', 'mask-noise': 'noise' };
const LABEL_OF: Record<MaskModifierKind, string> = {
  'mask-denoise': 'widgets.layers.maskFill.denoiseLimit',
  'mask-noise': 'widgets.layers.maskFill.noiseLevel',
};
const HELP_OF: Record<MaskModifierKind, string> = {
  'mask-denoise': 'widgets.layers.modifiers.denoiseHelp',
  'mask-noise': 'widgets.layers.modifiers.noiseHelp',
};

const formatPercent = (value: number): string => `${value}%`;

const magnitudeOf = (modifier: MaskModifier): number => ('level' in modifier ? modifier.level : modifier.limit);

export const MaskModifierSettings = ({
  engine,
  kind,
  layer,
}: {
  engine: (CanvasStructuralEngine & CanvasPreparedEngine) | null;
  kind: MaskModifierKind;
  layer: CanvasInpaintMaskLayerContract;
}) => {
  const { t } = useTranslation();
  const { cancel: cancelPreview, commit: commitPrepared, preview: previewStructural } = useStructuralPreview(engine);
  const field = FIELD_OF[kind];
  const modifier = layer[field];
  // Where a gesture started; outside render closures because a drag keeps the handlers it started with.
  const beforeRef = useRef<MaskModifier | null>(null);

  const configWith = useCallback(
    (base: MaskModifier, magnitude: number) =>
      ({
        layerType: 'inpaint_mask',
        [field]: field === 'noise' ? { ...base, level: magnitude } : { ...base, limit: magnitude },
      }) as const,
    [field]
  );

  const handleChange = useCallback(
    (percent: number) => {
      if (!modifier) {
        return;
      }
      if (
        previewStructural({
          config: configWith(modifier, percent / 100),
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        })
      ) {
        beforeRef.current ??= modifier;
      }
    },
    [configWith, layer.id, modifier, previewStructural]
  );

  const handleChangeEnd = useCallback(
    (percent: number) => {
      const before = beforeRef.current;
      beforeRef.current = null;
      if (!before) {
        return;
      }
      const next = percent / 100;
      if (next === magnitudeOf(before)) {
        cancelPreview({
          config: { layerType: 'inpaint_mask', [field]: before },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        });
        return;
      }
      // The commit changes only the magnitude: `isEnabled` stays live, so a
      // toggle landing mid-gesture is not silently reverted.
      const committed = { ...before, isEnabled: modifier?.isEnabled ?? before.isEnabled };
      commitPrepared(t(LABEL_OF[kind]), (model) =>
        model.prepare({
          before: { layerType: 'inpaint_mask', [field]: committed },
          config: configWith(committed, next),
          id: layer.id,
          type: 'patch-config',
        })
      );
    },
    [cancelPreview, commitPrepared, configWith, field, kind, layer.id, modifier?.isEnabled, t]
  );

  if (!modifier) {
    return null;
  }
  return (
    <ScrubberField
      defaultValue={Math.round(MASK_MODIFIER_DEFAULTS[field] * 100)}
      formatValue={formatPercent}
      helpText={t(HELP_OF[kind])}
      label={t(LABEL_OF[kind])}
      max={100}
      min={0}
      step={1}
      value={Math.round(magnitudeOf(modifier) * 100)}
      onChange={handleChange}
      onChangeEnd={handleChangeEnd}
    />
  );
};

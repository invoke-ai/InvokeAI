import type { CanvasDocumentModel, CanvasInpaintMaskLayerContract } from '@workbench/canvas-engine/api';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';

import { ScrubberField } from '@platform/ui/ScrubberField';
import {
  baselineConfig,
  type CanvasPreparedEngine,
  useStructuralPreview,
} from '@workbench/widgets/canvas/useStructuralCommit';
import { useCallback } from 'react';
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
  const { commit: commitPrepared, preview: previewStructural } = useStructuralPreview(engine);
  const field = FIELD_OF[kind];
  const modifier = layer[field];

  const liveModifier = useCallback(
    (model: CanvasDocumentModel | null | undefined): MaskModifier | null => {
      const live = model?.getLayer(layer.id);
      return live?.type === 'inpaint_mask' ? (live[field] ?? null) : null;
    },
    [field, layer.id]
  );
  const configWith = useCallback(
    (base: MaskModifier, magnitude: number) =>
      ({
        layerType: 'inpaint_mask',
        [field]: field === 'noise' ? { ...base, level: magnitude } : { ...base, limit: magnitude },
      }) as const,
    [field]
  );

  // Previews set only the magnitude on the live modifier (a drag keeps the handlers it started with).
  const handleChange = useCallback(
    (percent: number) => {
      const live = liveModifier(engine?.document.model());
      if (live) {
        previewStructural({ config: configWith(live, percent / 100), id: layer.id, type: 'updateCanvasLayerConfig' });
      }
    },
    [configWith, engine, layer.id, liveModifier, previewStructural]
  );

  // A gesture ending where it started prepares nothing, which drops its previews.
  const handleChangeEnd = useCallback(
    (percent: number) => {
      commitPrepared(t(LABEL_OF[kind]), (model, baseline) => {
        const live = liveModifier(model);
        // Removed mid-gesture: committing would bring it back.
        if (!live) {
          return { ids: [layer.id], status: 'missing' };
        }
        const config = configWith(live, percent / 100);
        return model.prepare({ before: baselineConfig(baseline, config), config, id: layer.id, type: 'patch-config' });
      });
    },
    [commitPrepared, configWith, kind, layer.id, liveModifier, t]
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

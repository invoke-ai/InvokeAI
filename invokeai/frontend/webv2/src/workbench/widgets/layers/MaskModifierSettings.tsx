import type { SliderValueChangeDetails } from '@chakra-ui/react';
import type { CanvasInpaintMaskLayerContract } from '@workbench/canvas-engine/api';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';

import { Stack, Text } from '@chakra-ui/react';
import { Field, Slider } from '@platform/ui';
import { type CanvasPreparedEngine, useStructuralPreview } from '@workbench/widgets/canvas/useStructuralCommit';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

/** Edit modifier magnitude here with one commit per gesture; tree rows own enabling and removal. */

type MaskModifierKind = 'mask-noise' | 'mask-denoise';

const FIELD_OF: Record<MaskModifierKind, 'noise' | 'denoise'> = { 'mask-denoise': 'denoise', 'mask-noise': 'noise' };
const LABEL_OF: Record<MaskModifierKind, string> = {
  'mask-denoise': 'widgets.layers.maskFill.denoiseLimit',
  'mask-noise': 'widgets.layers.maskFill.noiseLevel',
};
const HELP_OF: Record<MaskModifierKind, string> = {
  'mask-denoise': 'widgets.layers.modifiers.denoiseHelp',
  'mask-noise': 'widgets.layers.modifiers.noiseHelp',
};

const formatUnitPercent = (value: number): string => `${Math.round(value * 100)}%`;

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
  const beforeRef = useRef<typeof modifier | null>(null);

  const handleChange = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      const next = value[0];
      if (next === undefined || !Number.isFinite(next) || !modifier) {
        return;
      }
      const config = {
        layerType: 'inpaint_mask',
        [field]: field === 'noise' ? { ...modifier, level: next } : { ...modifier, limit: next },
      } as const;
      if (previewStructural({ config, id: layer.id, type: 'updateCanvasLayerConfig' })) {
        beforeRef.current ??= modifier;
      }
    },
    [previewStructural, field, layer.id, modifier]
  );

  const handleChangeEnd = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      const next = value[0];
      const before = beforeRef.current ?? modifier;
      beforeRef.current = null;
      if (next === undefined || !Number.isFinite(next) || !before || !modifier) {
        return;
      }
      // The commit changes only the magnitude: `isEnabled` stays live, so a
      // toggle landing mid-gesture is not silently reverted.
      const committed = { ...before, isEnabled: modifier.isEnabled };
      commitPrepared(t(LABEL_OF[kind]), (model) =>
        model.prepare({
          before: { layerType: 'inpaint_mask', [field]: committed },
          config: {
            layerType: 'inpaint_mask',
            [field]: field === 'noise' ? { ...committed, level: next } : { ...committed, limit: next },
          },
          id: layer.id,
          type: 'patch-config',
        })
      );
    },
    [commitPrepared, field, kind, layer.id, modifier, t]
  );

  const sliderValue = useMemo(
    () => [modifier ? ('level' in modifier ? modifier.level : modifier.limit) : 0],
    [modifier]
  );
  const sliderAria = useMemo(() => [t(LABEL_OF[kind])], [kind, t]);

  if (!modifier) {
    return null;
  }
  return (
    <Stack gap="2">
      <Field label={t(LABEL_OF[kind])}>
        <Slider
          aria-label={sliderAria}
          formatValue={formatUnitPercent}
          max={1}
          min={0}
          size="sm"
          step={0.01}
          value={sliderValue}
          withThumbTooltip
          onValueChange={handleChange}
          onValueChangeEnd={handleChangeEnd}
        />
      </Field>
      <Text color="fg.muted" fontSize="2xs">
        {t(HELP_OF[kind])}
      </Text>
    </Stack>
  );
};

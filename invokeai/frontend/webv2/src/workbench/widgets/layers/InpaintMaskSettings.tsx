import type { SelectValueChangeDetails } from '@chakra-ui/react';
import type { CanvasInpaintMaskLayerContract, CanvasMaskFillContract } from '@workbench/canvas-engine/api';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';

import { createListCollection, HStack, Stack } from '@chakra-ui/react';
import { Button, ColorPicker, Field, IconButton, Select, Tooltip } from '@platform/ui';
import { useNotify } from '@workbench/useNotify';
import { armMaskTintTarget } from '@workbench/widgets/canvas/color-system/maskTintTarget';
import { type ColorSamplerEngine, useColorSampler } from '@workbench/widgets/canvas/useColorSampler';
import {
  type CanvasPreparedEngine,
  reportMaskEdit,
  useStructuralPreview,
} from '@workbench/widgets/canvas/useStructuralCommit';
import { PaletteIcon } from 'lucide-react';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

/** The six mask fill styles, matching `CanvasMaskFillContract['style']` / legacy `zFillStyle`. */
const MASK_FILL_STYLES: readonly CanvasMaskFillContract['style'][] = [
  'solid',
  'grid',
  'crosshatch',
  'diagonal',
  'horizontal',
  'vertical',
];

interface InpaintMaskSettingsProps {
  engine: (CanvasStructuralEngine & CanvasPreparedEngine & ColorSamplerEngine) | null;
  layer: CanvasInpaintMaskLayerContract;
}

/**
 * Properties owns fill/style and invert; noise/denoise modifiers have tree-child editors. Fill uses undoable
 * config patches, invert uses engine pixels.
 */
export const InpaintMaskSettings = ({ engine, layer }: InpaintMaskSettingsProps) => {
  const { t } = useTranslation();
  const { commit: commitPrepared, preview: previewStructural } = useStructuralPreview(engine);
  const sampleColor = useColorSampler(engine);
  const fillBeforeRef = useRef<CanvasMaskFillContract | null>(null);

  const fill = layer.mask.fill;

  const styleCollection = useMemo(
    () =>
      createListCollection({
        items: MASK_FILL_STYLES.map((style) => ({
          label: t(`widgets.layers.maskFill.styles.${style}`),
          value: style,
        })),
      }),
    [t]
  );

  const commitFill = useCallback(
    (next: CanvasMaskFillContract, before: CanvasMaskFillContract) => {
      commitPrepared(t('widgets.layers.maskFill.fill'), (model) =>
        model.prepare({
          before: { layerType: 'inpaint_mask', mask: { fill: before } },
          config: { layerType: 'inpaint_mask', mask: { fill: next } },
          id: layer.id,
          type: 'patch-config',
        })
      );
    },
    [commitPrepared, layer.id, t]
  );

  const handleColorChange = useCallback(
    (hex: string) => {
      if (
        !previewStructural({
          config: { layerType: 'inpaint_mask', mask: { fill: { ...fill, color: hex } } },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        })
      ) {
        return;
      }
      if (fillBeforeRef.current === null) {
        fillBeforeRef.current = fill;
      }
    },
    [previewStructural, fill, layer.id]
  );

  const handleArmTint = useCallback(() => armMaskTintTarget(layer.id), [layer.id]);
  const handleColorChangeEnd = useCallback(
    (hex: string) => {
      const before = fillBeforeRef.current ?? fill;
      fillBeforeRef.current = null;
      commitFill({ ...before, color: hex }, before);
    },
    [commitFill, fill]
  );

  const handleStyleChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const style = value[0] as CanvasMaskFillContract['style'] | undefined;
      if (style && style !== fill.style) {
        commitFill({ ...fill, style }, fill);
      }
    },
    [commitFill, fill]
  );

  const notify = useNotify();
  const handleInvert = useCallback(() => {
    if (engine) {
      reportMaskEdit(engine.layers.invertMask(layer.id), notify.error, t);
    }
  }, [engine, layer.id, notify, t]);

  const styleValue = useMemo(() => [fill.style], [fill.style]);
  const colorAria = t('widgets.layers.maskFill.color');

  return (
    <Stack gap="2">
      <HStack gap="2">
        <Field flexShrink="0" label={t('widgets.layers.maskFill.color')}>
          <ColorPicker
            aria-label={colorAria}
            value={fill.color}
            onSampleColor={sampleColor}
            onValueChange={handleColorChange}
            onValueChangeEnd={handleColorChangeEnd}
          />
        </Field>
        <Tooltip content={t('widgets.layers.maskFill.editInColorPane')}>
          <IconButton
            aria-label={t('widgets.layers.maskFill.editInColorPane')}
            alignSelf="flex-end"
            color="fg.muted"
            size="2xs"
            variant="ghost"
            onClick={handleArmTint}
          >
            <PaletteIcon size={14} />
          </IconButton>
        </Tooltip>
        <Field flex="1" label={t('widgets.layers.maskFill.style')} minW="0">
          <Select
            aria-label={t('widgets.layers.maskFill.style')}
            collection={styleCollection}
            positioning={SELECT_POSITIONING}
            size="xs"
            value={styleValue}
            valueText={t(`widgets.layers.maskFill.styles.${fill.style}`)}
            onValueChange={handleStyleChange}
          />
        </Field>
      </HStack>
      <Button disabled={!engine} size="xs" variant="outline" onClick={handleInvert}>
        {t('widgets.layers.maskFill.invert')}
      </Button>
    </Stack>
  );
};

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: true } as const;

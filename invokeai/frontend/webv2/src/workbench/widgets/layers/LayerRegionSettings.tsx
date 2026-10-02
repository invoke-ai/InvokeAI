import type { SelectValueChangeDetails } from '@chakra-ui/react';
import type { CanvasMaskFillContract, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/api';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';

import { createListCollection, HStack, Stack, Text } from '@chakra-ui/react';
import { ColorPicker, Field, Select } from '@platform/ui';
import { type ColorSamplerEngine, useColorSampler } from '@workbench/widgets/canvas/useColorSampler';
import { type CanvasPreparedEngine, useStructuralPreview } from '@workbench/widgets/canvas/useStructuralCommit';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

const MASK_FILL_STYLES: readonly CanvasMaskFillContract['style'][] = [
  'solid',
  'grid',
  'crosshatch',
  'diagonal',
  'horizontal',
  'vertical',
];

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: true } as const;

interface LayerRegionSettingsProps {
  engine: (CanvasStructuralEngine & CanvasPreparedEngine & ColorSamplerEngine) | null;
  layer: CanvasRasterLayerContractV2;
}

/** The regenerate region's dedicated Properties editor: its overlay fill color and style. */
export const LayerRegionSettings = ({ engine, layer }: LayerRegionSettingsProps) => {
  const { t } = useTranslation();
  const { commit: commitPrepared, preview: previewStructural } = useStructuralPreview(engine);
  const sampleColor = useColorSampler(engine);
  const fillBeforeRef = useRef<CanvasMaskFillContract | null>(null);
  const region = layer.inpaint;

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
      if (!region) {
        return;
      }
      commitPrepared(t('widgets.layers.maskFill.fill'), (model) =>
        model.prepare({
          before: { inpaint: { ...region, fill: before }, layerType: 'raster' },
          config: { inpaint: { ...region, fill: next }, layerType: 'raster' },
          id: layer.id,
          type: 'patch-config',
        })
      );
    },
    [commitPrepared, layer.id, region, t]
  );

  const handleColorChange = useCallback(
    (hex: string) => {
      if (!region) {
        return;
      }
      if (
        !previewStructural({
          config: {
            inpaint: { ...region, fill: { ...region.fill, color: hex } },
            layerType: 'raster',
          },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        })
      ) {
        return;
      }
      fillBeforeRef.current ??= region.fill;
    },
    [previewStructural, layer.id, region]
  );

  const handleColorChangeEnd = useCallback(
    (hex: string) => {
      if (!region) {
        return;
      }
      const before = fillBeforeRef.current ?? region.fill;
      fillBeforeRef.current = null;
      commitFill({ ...before, color: hex }, before);
    },
    [commitFill, region]
  );

  const handleStyleChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const style = value[0] as CanvasMaskFillContract['style'] | undefined;
      if (region && style && style !== region.fill.style) {
        commitFill({ ...region.fill, style }, region.fill);
      }
    },
    [commitFill, region]
  );

  const styleValue = useMemo(() => [region?.fill.style ?? 'solid'], [region?.fill.style]);
  if (!region) {
    return null;
  }
  const fill = region.fill;

  return (
    <Stack gap="2">
      <HStack gap="2">
        <Field flexShrink="0" label={t('widgets.layers.maskFill.color')}>
          <ColorPicker
            aria-label={t('widgets.layers.maskFill.color')}
            value={fill.color}
            onSampleColor={sampleColor}
            onValueChange={handleColorChange}
            onValueChangeEnd={handleColorChangeEnd}
          />
        </Field>
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
      <Text color="fg.muted" fontSize="xs">
        {t('widgets.layers.modifiers.regionHint')}
      </Text>
    </Stack>
  );
};

import { Flex, FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InpaintMaskDeleteModifierButton } from 'features/controlLayers/components/InpaintMask/InpaintMaskDeleteModifierButton';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rasterLayerGlobalCompositeOperationChanged } from 'features/controlLayers/store/canvasSlice';
import type { CompositeOperation } from 'features/controlLayers/store/compositeOperations';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const RasterLayerCompositeOperationSettings = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();

  const layer = useAppSelector((s) =>
    s.canvas.present.rasterLayers.entities.find((e: CanvasRasterLayerState) => e.id === entityIdentifier.id)
  );

  const showSettings = useMemo(() => {
    return layer?.globalCompositeOperation !== undefined;
  }, [layer]);

  const currentOperation = useMemo(() => {
    return layer?.globalCompositeOperation ?? 'source-over';
  }, [layer]);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value as CompositeOperation;
      dispatch(rasterLayerGlobalCompositeOperationChanged({ entityIdentifier, globalCompositeOperation: value }));
    },
    [dispatch, entityIdentifier]
  );

  const onDelete = useCallback(() => {
    dispatch(
      rasterLayerGlobalCompositeOperationChanged({
        entityIdentifier,
        globalCompositeOperation: undefined,
      })
    );
  }, [dispatch, entityIdentifier]);

  if (!showSettings) {
    return null;
  }

  const COLOR_BLEND_MODES: CompositeOperation[] = [
    'multiply',
    'screen',
    'darken',
    'lighten',
    'color-dodge',
    'color-burn',
    'hard-light',
    'soft-light',
    'difference',
    'hue',
    'saturation',
    'color',
    'luminosity',
  ];

  return (
    <Flex px={2} pb={2}>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.compositeOperation.label')}</FormLabel>
        <Flex alignItems="center" mb={1}>
          <Select value={currentOperation} onChange={onChange} size="sm" flex={1} mr={2}>
            {COLOR_BLEND_MODES.map((op) => (
              <option key={op} value={op}>
                {op}
              </option>
            ))}
          </Select>
          <InpaintMaskDeleteModifierButton onDelete={onDelete} />
        </Flex>
      </FormControl>
    </Flex>
  );
});

RasterLayerCompositeOperationSettings.displayName = 'RasterLayerCompositeOperationSettings';

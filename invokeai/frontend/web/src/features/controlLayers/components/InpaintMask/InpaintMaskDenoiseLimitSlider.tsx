import { Flex, Slider, SliderFilledTrack, SliderThumb, SliderTrack, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InpaintMaskDeleteModifierButton } from 'features/controlLayers/components/InpaintMask/InpaintMaskDeleteModifierButton';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  inpaintMaskDenoiseLimitChanged,
  inpaintMaskDenoiseLimitDeleted,
} from 'features/controlLayers/store/canvasInstanceSlice';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const InpaintMaskDenoiseLimitSlider = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const selectDenoiseLimit = useMemo(
    () =>
      createSelector(
        selectCanvasSlice,
        (canvas) => canvas ? selectEntityOrThrow(canvas, entityIdentifier, 'InpaintMaskDenoiseLimitSlider').denoiseLimit : undefined
      ),
    [entityIdentifier]
  );
  const denoiseLimit = useAppSelector(selectDenoiseLimit);

  const handleDenoiseLimitChange = useCallback(
    (value: number) => {
      dispatch(inpaintMaskDenoiseLimitChanged({ entityIdentifier, denoiseLimit: value }));
    },
    [dispatch, entityIdentifier]
  );

  const onDeleteDenoiseLimit = useCallback(() => {
    dispatch(inpaintMaskDenoiseLimitDeleted({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  if (denoiseLimit === undefined) {
    return null;
  }

  return (
    <Flex direction="column" gap={1} w="full" px={2} pb={2}>
      <Flex justifyContent="space-between" w="full" alignItems="center">
        <Text fontSize="sm">{t('controlLayers.denoiseLimit')}</Text>
        <Flex alignItems="center" gap={1}>
          <Text fontSize="sm">{denoiseLimit.toFixed(2)}</Text>
          <InpaintMaskDeleteModifierButton onDelete={onDeleteDenoiseLimit} />
        </Flex>
      </Flex>
      <Slider
        aria-label={t('controlLayers.denoiseLimit')}
        value={denoiseLimit}
        min={0}
        max={1}
        step={0.01}
        onChange={handleDenoiseLimitChange}
      >
        <SliderTrack>
          <SliderFilledTrack />
        </SliderTrack>
        <SliderThumb />
      </Slider>
    </Flex>
  );
});

InpaintMaskDenoiseLimitSlider.displayName = 'InpaintMaskDenoiseLimitSlider';

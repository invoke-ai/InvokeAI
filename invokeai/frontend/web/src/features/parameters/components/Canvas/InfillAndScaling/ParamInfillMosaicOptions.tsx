import { Box, CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIColorPicker from 'common/components/IAIColorPicker';
import {
  setInfillMosaicMaxColor,
  setInfillMosaicMinColor,
  setInfillMosaicTileHeight,
  setInfillMosaicTileWidth,
} from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import type { RgbaColor } from 'react-colorful';
import { useTranslation } from 'react-i18next';

const ParamInfillMosaicTileSize = () => {
  const dispatch = useAppDispatch();

  const infillMosaicTileWidth = useAppSelector((s) => s.generation.infillMosaicTileWidth);
  const infillMosaicTileHeight = useAppSelector((s) => s.generation.infillMosaicTileHeight);
  const infillMosaicMinColor = useAppSelector((s) => s.generation.infillMosaicMinColor);
  const infillMosaicMaxColor = useAppSelector((s) => s.generation.infillMosaicMaxColor);
  const infillMethod = useAppSelector((s) => s.generation.infillMethod);

  const { t } = useTranslation();

  const handleInfillMosaicTileWidthChange = useCallback(
    (v: number) => {
      dispatch(setInfillMosaicTileWidth(v));
    },
    [dispatch]
  );

  const handleInfillMosaicTileHeightChange = useCallback(
    (v: number) => {
      dispatch(setInfillMosaicTileHeight(v));
    },
    [dispatch]
  );

  const handleInfillMosaicMinColor = useCallback(
    (v: RgbaColor) => {
      dispatch(setInfillMosaicMinColor(v));
    },
    [dispatch]
  );

  const handleInfillMosaicMaxColor = useCallback(
    (v: RgbaColor) => {
      dispatch(setInfillMosaicMaxColor(v));
    },
    [dispatch]
  );

  return (
    <Flex flexDir="column" gap={4}>
      <FormControl isDisabled={infillMethod !== 'mosaic'}>
        <FormLabel>{t('parameters.infillMosaicTileWidth')}</FormLabel>
        <CompositeSlider
          min={8}
          max={256}
          value={infillMosaicTileWidth}
          defaultValue={64}
          onChange={handleInfillMosaicTileWidthChange}
          step={8}
          fineStep={8}
          marks
        />
        <CompositeNumberInput
          min={8}
          max={1024}
          value={infillMosaicTileWidth}
          defaultValue={64}
          onChange={handleInfillMosaicTileWidthChange}
          step={8}
          fineStep={8}
        />
      </FormControl>
      <FormControl isDisabled={infillMethod !== 'mosaic'}>
        <FormLabel>{t('parameters.infillMosaicTileHeight')}</FormLabel>
        <CompositeSlider
          min={8}
          max={256}
          value={infillMosaicTileHeight}
          defaultValue={64}
          onChange={handleInfillMosaicTileHeightChange}
          step={8}
          fineStep={8}
          marks
        />
        <CompositeNumberInput
          min={8}
          max={1024}
          value={infillMosaicTileHeight}
          defaultValue={64}
          onChange={handleInfillMosaicTileHeightChange}
          step={8}
          fineStep={8}
        />
      </FormControl>
      <FormControl isDisabled={infillMethod !== 'mosaic'}>
        <FormLabel>{t('parameters.infillMosaicMinColor')}</FormLabel>
        <Box w="full" pt={2} pb={2}>
          <IAIColorPicker color={infillMosaicMinColor} onChange={handleInfillMosaicMinColor} />
        </Box>
      </FormControl>
      <FormControl isDisabled={infillMethod !== 'mosaic'}>
        <FormLabel>{t('parameters.infillMosaicMaxColor')}</FormLabel>
        <Box w="full" pt={2} pb={2}>
          <IAIColorPicker color={infillMosaicMaxColor} onChange={handleInfillMosaicMaxColor} />
        </Box>
      </FormControl>
    </Flex>
  );
};

export default memo(ParamInfillMosaicTileSize);

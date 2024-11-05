import { Badge, Box, CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import WavyLine from 'common/components/WavyLine';
import { selectImg2imgStrength, setImg2imgStrength } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { selectImg2imgStrengthConfig } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const marks = [0, 0.5, 1];

export const ParamDenoisingStrength = memo(() => {
  const img2imgStrength = useAppSelector(selectImg2imgStrength);
  const dispatch = useAppDispatch();
  const isEnabled = useAppSelector(
    (s) =>
      selectCanvasSlice(s).rasterLayers.entities.length > 0 &&
      selectCanvasSlice(s).rasterLayers.entities.some((layer) => layer.isEnabled)
  );

  const onChange = useCallback(
    (v: number) => {
      dispatch(setImg2imgStrength(v));
    },
    [dispatch]
  );

  const config = useAppSelector(selectImg2imgStrengthConfig);
  const { t } = useTranslation();

  return (
    <FormControl isDisabled={!isEnabled} py={2} justifyContent="space-between">
      <Flex gap={3} alignItems="center">
        <InformationalPopover feature="paramDenoisingStrength">
          <FormLabel mr={0}>{`${t('parameters.denoisingStrength')}`}</FormLabel>
        </InformationalPopover>
        <Box position="relative" opacity={!isEnabled ? 0.5 : 1}>
          <WavyLine waviness={img2imgStrength * 10} />
        </Box>
      </Flex>
      {isEnabled ? (
        <>
          <CompositeSlider
            step={config.coarseStep}
            fineStep={config.fineStep}
            min={config.sliderMin}
            max={config.sliderMax}
            defaultValue={config.initial}
            onChange={onChange}
            value={img2imgStrength}
            marks={marks}
          />
          <CompositeNumberInput
            step={config.coarseStep}
            fineStep={config.fineStep}
            min={config.numberInputMin}
            max={config.numberInputMax}
            defaultValue={config.initial}
            onChange={onChange}
            value={img2imgStrength}
          />
        </>
      ) : (
        <Flex justifySelf="flex-end">
          <Badge opacity="0.6">
            {t('common.disabled')} - {t('parameters.noRasterLayers')}
          </Badge>
        </Flex>
      )}
    </FormControl>
  );
});

ParamDenoisingStrength.displayName = 'ParamDenoisingStrength';

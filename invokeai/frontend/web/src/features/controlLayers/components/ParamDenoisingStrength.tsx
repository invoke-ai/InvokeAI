import {
  Badge,
  Box,
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  FormLabel,
  useToken,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import WavyLine from 'common/components/WavyLine';
import { selectImg2imgStrength, setImg2imgStrength } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { selectImg2imgStrengthConfig } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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

  const [invokeBlue300] = useToken('colors', ['invokeBlue.300']);

  return (
    <FormControl isDisabled={!isEnabled} p={1} justifyContent="space-between" h={8}>
      <Flex gap={3} alignItems="center">
        <InformationalPopover feature="paramDenoisingStrength">
          <FormLabel mr={0}>{`${t('parameters.denoisingStrength')}`}</FormLabel>
        </InformationalPopover>
        {isEnabled && (
            <WavyLine amplitude={img2imgStrength * 10} stroke={invokeBlue300} strokeWidth={1} width={40} height={14} />
        )}
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
          />
          <CompositeNumberInput
            step={config.coarseStep}
            fineStep={config.fineStep}
            min={config.numberInputMin}
            max={config.numberInputMax}
            defaultValue={config.initial}
            onChange={onChange}
            value={img2imgStrength}
            variant="outline"
          />
        </>
      ) : (
        <Flex alignItems="center">
          <Badge opacity="0.6">
            {t('common.disabled')} - {t('parameters.noRasterLayers')}
          </Badge>
        </Flex>
      )}
    </FormControl>
  );
});

ParamDenoisingStrength.displayName = 'ParamDenoisingStrength';

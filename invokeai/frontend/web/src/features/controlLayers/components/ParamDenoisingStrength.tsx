import {
  Badge,
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  FormLabel,
  useToken,
} from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import WavyLine from 'common/components/WavyLine';
import { selectImg2imgStrength, setImg2imgStrength } from 'features/controlLayers/store/paramsSlice';
import { selectActiveRasterLayerEntities } from 'features/controlLayers/store/selectors';
import { selectImg2imgStrengthConfig } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectHasRasterLayersWithContent = createSelector(
  selectActiveRasterLayerEntities,
  (entities) => entities.length > 0
);

export const ParamDenoisingStrength = memo(() => {
  const img2imgStrength = useAppSelector(selectImg2imgStrength);
  const dispatch = useAppDispatch();
  const hasRasterLayersWithContent = useAppSelector(selectHasRasterLayersWithContent);

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
    <FormControl isDisabled={!hasRasterLayersWithContent} p={1} justifyContent="space-between" h={8}>
      <Flex gap={3} alignItems="center">
        <InformationalPopover feature="paramDenoisingStrength">
          <FormLabel mr={0}>{`${t('parameters.denoisingStrength')}`}</FormLabel>
        </InformationalPopover>
        {hasRasterLayersWithContent && (
          <WavyLine amplitude={img2imgStrength * 10} stroke={invokeBlue300} strokeWidth={1} width={40} height={14} />
        )}
      </Flex>
      {hasRasterLayersWithContent ? (
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
          <Badge opacity="0.6">{t('parameters.disabledNoRasterContent')}</Badge>
        </Flex>
      )}
    </FormControl>
  );
});

ParamDenoisingStrength.displayName = 'ParamDenoisingStrength';

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
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useSelectedModelConfig } from 'services/api/hooks/useSelectedModelConfig';
import { isFluxFillMainModelModelConfig } from 'services/api/types';

const CONSTRAINTS = {
  initial: 0.7,
  sliderMin: 0,
  sliderMax: 1,
  numberInputMin: 0,
  numberInputMax: 1,
  fineStep: 0.01,
  coarseStep: 0.05,
};

const selectHasRasterLayersWithContent = createSelector(
  selectActiveRasterLayerEntities,
  (entities) => entities.length > 0
);

export const ParamDenoisingStrength = memo(() => {
  const img2imgStrength = useAppSelector(selectImg2imgStrength);
  const dispatch = useAppDispatch();
  const hasRasterLayersWithContent = useAppSelector(selectHasRasterLayersWithContent);
  const selectedModelConfig = useSelectedModelConfig();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setImg2imgStrength(v));
    },
    [dispatch]
  );

  const { t } = useTranslation();

  const [invokeBlue300] = useToken('colors', ['invokeBlue.300']);

  const isDisabled = useMemo(() => {
    if (!hasRasterLayersWithContent) {
      // Denoising strength does nothing if there are no raster layers w/ content
      return true;
    }
    if (selectedModelConfig && isFluxFillMainModelModelConfig(selectedModelConfig)) {
      // Denoising strength is ignored by FLUX Fill, which is indicated by the variant being 'inpaint'
      return true;
    }
    return false;
  }, [hasRasterLayersWithContent, selectedModelConfig]);

  return (
    <FormControl isDisabled={isDisabled} p={1} justifyContent="space-between" h={8}>
      <Flex gap={3} alignItems="center">
        <InformationalPopover feature="paramDenoisingStrength">
          <FormLabel mr={0}>{`${t('parameters.denoisingStrength')}`}</FormLabel>
        </InformationalPopover>
        {hasRasterLayersWithContent && (
          <WavyLine amplitude={img2imgStrength * 10} stroke={invokeBlue300} strokeWidth={1} width={40} height={14} />
        )}
      </Flex>
      {!isDisabled ? (
        <>
          <CompositeSlider
            step={CONSTRAINTS.coarseStep}
            fineStep={CONSTRAINTS.fineStep}
            min={CONSTRAINTS.sliderMin}
            max={CONSTRAINTS.sliderMax}
            defaultValue={CONSTRAINTS.initial}
            onChange={onChange}
            value={img2imgStrength}
          />
          <CompositeNumberInput
            step={CONSTRAINTS.coarseStep}
            fineStep={CONSTRAINTS.fineStep}
            min={CONSTRAINTS.numberInputMin}
            max={CONSTRAINTS.numberInputMax}
            defaultValue={CONSTRAINTS.initial}
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

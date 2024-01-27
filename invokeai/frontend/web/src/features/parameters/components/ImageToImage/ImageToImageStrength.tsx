import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { setImg2imgStrength } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const marks = [0, 0.5, 1];

const ImageToImageStrength = () => {
  const img2imgStrength = useAppSelector((s) => s.generation.img2imgStrength);
  const initial = useAppSelector((s) => s.config.sd.img2imgStrength.initial);
  const sliderMin = useAppSelector((s) => s.config.sd.img2imgStrength.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.img2imgStrength.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.img2imgStrength.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.img2imgStrength.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.img2imgStrength.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.img2imgStrength.fineStep);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback((v: number) => dispatch(setImg2imgStrength(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="paramDenoisingStrength">
        <FormLabel>{`${t('parameters.denoisingStrength')}`}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        step={coarseStep}
        fineStep={fineStep}
        min={sliderMin}
        max={sliderMax}
        onChange={handleChange}
        value={img2imgStrength}
        defaultValue={initial}
        marks={marks}
      />
      <CompositeNumberInput
        step={coarseStep}
        fineStep={fineStep}
        min={numberInputMin}
        max={numberInputMax}
        onChange={handleChange}
        value={img2imgStrength}
        defaultValue={initial}
      />
    </FormControl>
  );
};

export default memo(ImageToImageStrength);

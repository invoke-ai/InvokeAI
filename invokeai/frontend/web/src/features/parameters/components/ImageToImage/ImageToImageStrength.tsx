import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setImg2imgStrength } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const marks = [0, 0.5, 1];

const ImageToImageStrength = () => {
  const img2imgStrength = useAppSelector((s) => s.generation.img2imgStrength);
  const initial = useAppSelector((s) => s.config.sd.img2imgStrength.initial);
  const min = useAppSelector((s) => s.config.sd.img2imgStrength.min);
  const sliderMax = useAppSelector(
    (s) => s.config.sd.img2imgStrength.sliderMax
  );
  const inputMax = useAppSelector((s) => s.config.sd.img2imgStrength.inputMax);
  const coarseStep = useAppSelector(
    (s) => s.config.sd.img2imgStrength.coarseStep
  );
  const fineStep = useAppSelector((s) => s.config.sd.img2imgStrength.fineStep);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => dispatch(setImg2imgStrength(v)),
    [dispatch]
  );

  return (
    <InvControl
      label={`${t('parameters.denoisingStrength')}`}
      feature="paramDenoisingStrength"
    >
      <InvSlider
        step={coarseStep}
        fineStep={fineStep}
        min={min}
        max={sliderMax}
        onChange={handleChange}
        value={img2imgStrength}
        defaultValue={initial}
        marks={marks}
        withNumberInput
        numberInputMax={inputMax}
      />
    </InvControl>
  );
};

export default memo(ImageToImageStrength);

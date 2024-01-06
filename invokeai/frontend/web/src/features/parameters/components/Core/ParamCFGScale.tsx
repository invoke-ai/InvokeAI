import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setCfgScale } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCFGScale = () => {
  const cfgScale = useAppSelector((s) => s.generation.cfgScale);
  const min = useAppSelector((s) => s.config.sd.guidance.min);
  const inputMax = useAppSelector((s) => s.config.sd.guidance.inputMax);
  const sliderMax = useAppSelector((s) => s.config.sd.guidance.sliderMax);
  const coarseStep = useAppSelector((s) => s.config.sd.guidance.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.guidance.fineStep);
  const initial = useAppSelector((s) => s.config.sd.guidance.initial);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const marks = useMemo(
    () => [min, Math.floor(sliderMax / 2), sliderMax],
    [sliderMax, min]
  );
  const onChange = useCallback(
    (v: number) => dispatch(setCfgScale(v)),
    [dispatch]
  );

  return (
    <InvControl label={t('parameters.cfgScale')} feature="paramCFGScale">
      <InvSlider
        value={cfgScale}
        defaultValue={initial}
        min={min}
        max={sliderMax}
        step={coarseStep}
        fineStep={fineStep}
        onChange={onChange}
        withNumberInput
        marks={marks}
        numberInputMax={inputMax}
      />
    </InvControl>
  );
};

export default memo(ParamCFGScale);

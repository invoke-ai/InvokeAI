import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setCfgRescaleMultiplier } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCFGRescaleMultiplier = () => {
  const cfgRescaleMultiplier = useAppSelector(
    (s) => s.generation.cfgRescaleMultiplier
  );
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => dispatch(setCfgRescaleMultiplier(v)),
    [dispatch]
  );

  return (
    <InvControl
      label={t('parameters.cfgRescaleMultiplier')}
      feature="paramCFGRescaleMultiplier"
    >
      <InvSlider
        value={cfgRescaleMultiplier}
        defaultValue={0}
        min={0}
        max={0.99}
        step={0.1}
        fineStep={0.01}
        onChange={handleChange}
        withNumberInput
        marks
      />
    </InvControl>
  );
};

export default memo(ParamCFGRescaleMultiplier);

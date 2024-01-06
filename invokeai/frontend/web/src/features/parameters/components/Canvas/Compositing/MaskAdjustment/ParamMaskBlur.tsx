import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setMaskBlur } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamMaskBlur = () => {
  const dispatch = useAppDispatch();
  const maskBlur = useAppSelector((s) => s.generation.maskBlur);
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setMaskBlur(v));
    },
    [dispatch]
  );

  return (
    <InvControl label={t('parameters.maskBlur')} feature="compositingBlur">
      <InvSlider
        min={0}
        max={64}
        value={maskBlur}
        defaultValue={16}
        onChange={handleChange}
        marks
        withNumberInput
        numberInputMax={512}
      />
    </InvControl>
  );
};

export default memo(ParamMaskBlur);

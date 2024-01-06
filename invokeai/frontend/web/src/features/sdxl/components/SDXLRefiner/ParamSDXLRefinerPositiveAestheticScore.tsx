import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setRefinerPositiveAestheticScore } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerPositiveAestheticScore = () => {
  const refinerPositiveAestheticScore = useAppSelector(
    (state) => state.sdxl.refinerPositiveAestheticScore
  );
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => dispatch(setRefinerPositiveAestheticScore(v)),
    [dispatch]
  );

  return (
    <InvControl label={t('sdxl.posAestheticScore')}>
      <InvSlider
        step={0.5}
        min={1}
        max={10}
        fineStep={0.1}
        onChange={handleChange}
        value={refinerPositiveAestheticScore}
        defaultValue={6}
        withNumberInput
        marks
      />
    </InvControl>
  );
};

export default memo(ParamSDXLRefinerPositiveAestheticScore);

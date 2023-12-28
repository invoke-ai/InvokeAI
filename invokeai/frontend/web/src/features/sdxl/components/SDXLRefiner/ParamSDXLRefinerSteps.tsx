import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setRefinerSteps } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerSteps = () => {
  const refinerSteps = useAppSelector((state) => state.sdxl.refinerSteps);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const handleChange = useCallback(
    (v: number) => {
      dispatch(setRefinerSteps(v));
    },
    [dispatch]
  );

  return (
    <InvControl label={t('sdxl.steps')}>
      <InvSlider
        min={1}
        max={100}
        step={1}
        onChange={handleChange}
        value={refinerSteps}
        withNumberInput
      />
    </InvControl>
  );
};

export default memo(ParamSDXLRefinerSteps);

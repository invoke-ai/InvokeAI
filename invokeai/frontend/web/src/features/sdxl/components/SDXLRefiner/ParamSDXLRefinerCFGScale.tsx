import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setRefinerCFGScale } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerCFGScale = () => {
  const refinerCFGScale = useAppSelector((state) => state.sdxl.refinerCFGScale);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => dispatch(setRefinerCFGScale(v)),
    [dispatch]
  );

  return (
    <InvControl label={t('sdxl.cfgScale')}>
      <InvSlider
        value={refinerCFGScale}
        min={1}
        max={200}
        step={0.5}
        fineStep={0.1}
        onChange={handleChange}
        withNumberInput
      />
    </InvControl>
  );
};

export default memo(ParamSDXLRefinerCFGScale);

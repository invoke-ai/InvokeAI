import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { setInfillPatchmatchDownscaleSize } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamInfillPatchmatchDownscaleSize = () => {
  const dispatch = useAppDispatch();
  const infillMethod = useAppSelector((s) => s.generation.infillMethod);
  const infillPatchmatchDownscaleSize = useAppSelector(
    (s) => s.generation.infillPatchmatchDownscaleSize
  );

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setInfillPatchmatchDownscaleSize(v));
    },
    [dispatch]
  );

  return (
    <InvControl
      isDisabled={infillMethod !== 'patchmatch'}
      label={t('parameters.patchmatchDownScaleSize')}
    >
      <InvSlider
        min={1}
        max={10}
        value={infillPatchmatchDownscaleSize}
        defaultValue={1}
        onChange={handleChange}
        withNumberInput
        marks
      />
    </InvControl>
  );
};

export default memo(ParamInfillPatchmatchDownscaleSize);

import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setRefinerPositiveAestheticScore } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerPositiveAestheticScore = () => {
  const refinerPositiveAestheticScore = useAppSelector((s) => s.sdxl.refinerPositiveAestheticScore);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback((v: number) => dispatch(setRefinerPositiveAestheticScore(v)), [dispatch]);

  return (
    <FormControl>
      <FormLabel>{t('sdxl.posAestheticScore')}</FormLabel>
      <CompositeSlider
        step={0.5}
        min={1}
        max={10}
        fineStep={0.1}
        onChange={handleChange}
        value={refinerPositiveAestheticScore}
        defaultValue={6}
        marks
      />
      <CompositeNumberInput
        step={0.5}
        min={1}
        max={10}
        fineStep={0.1}
        onChange={handleChange}
        value={refinerPositiveAestheticScore}
        defaultValue={6}
      />
    </FormControl>
  );
};

export default memo(ParamSDXLRefinerPositiveAestheticScore);

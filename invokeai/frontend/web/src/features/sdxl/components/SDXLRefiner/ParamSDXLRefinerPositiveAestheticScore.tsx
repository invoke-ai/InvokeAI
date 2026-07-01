import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectRefinerPositiveAestheticScore,
  setRefinerPositiveAestheticScore,
} from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerPositiveAestheticScore = () => {
  const refinerPositiveAestheticScore = useAppSelector(selectRefinerPositiveAestheticScore);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback((v: number) => dispatch(setRefinerPositiveAestheticScore(v)), [dispatch]);

  return (
    <FormControl>
      <InformationalPopover feature="refinerPositiveAestheticScore">
        <FormLabel>{t('sdxl.posAestheticScore')}</FormLabel>
      </InformationalPopover>
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

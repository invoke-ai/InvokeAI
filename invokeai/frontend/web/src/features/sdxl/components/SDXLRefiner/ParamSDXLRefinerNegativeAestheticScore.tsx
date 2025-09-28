import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectRefinerNegativeAestheticScore,
  setRefinerNegativeAestheticScore,
  useParamsDispatch,
} from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerNegativeAestheticScore = () => {
  const refinerNegativeAestheticScore = useAppSelector(selectRefinerNegativeAestheticScore);

  const dispatchParams = useParamsDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => dispatchParams(setRefinerNegativeAestheticScore, v),
    [dispatchParams]
  );

  return (
    <FormControl>
      <InformationalPopover feature="refinerNegativeAestheticScore">
        <FormLabel>{t('sdxl.negAestheticScore')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        min={1}
        max={10}
        step={0.5}
        fineStep={0.1}
        onChange={handleChange}
        value={refinerNegativeAestheticScore}
        defaultValue={2.5}
        marks
      />
      <CompositeNumberInput
        min={1}
        max={10}
        step={0.5}
        fineStep={0.1}
        onChange={handleChange}
        value={refinerNegativeAestheticScore}
        defaultValue={2.5}
      />
    </FormControl>
  );
};

export default memo(ParamSDXLRefinerNegativeAestheticScore);

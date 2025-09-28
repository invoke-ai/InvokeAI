import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectRefinerPositiveAestheticScore,
  setRefinerPositiveAestheticScore,
  useParamsDispatch,
} from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerPositiveAestheticScore = () => {
  const refinerPositiveAestheticScore = useAppSelector(selectRefinerPositiveAestheticScore);
  const dispatchParams = useParamsDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => dispatchParams(setRefinerPositiveAestheticScore, v),
    [dispatchParams]
  );

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

import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectRefinerStart, setRefinerStart, useParamsDispatch } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSDXLRefinerStart = () => {
  const refinerStart = useAppSelector(selectRefinerStart);
  const dispatchParams = useParamsDispatch();
  const handleChange = useCallback((v: number) => dispatchParams(setRefinerStart, v), [dispatchParams]);
  const { t } = useTranslation();

  return (
    <FormControl>
      <InformationalPopover feature="refinerStart">
        <FormLabel>{t('sdxl.refinerStart')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        step={0.01}
        min={0}
        max={1}
        onChange={handleChange}
        defaultValue={0.8}
        value={refinerStart}
        marks
      />
      <CompositeNumberInput
        step={0.01}
        min={0}
        max={1}
        onChange={handleChange}
        defaultValue={0.8}
        value={refinerStart}
      />
    </FormControl>
  );
};

export default memo(ParamSDXLRefinerStart);

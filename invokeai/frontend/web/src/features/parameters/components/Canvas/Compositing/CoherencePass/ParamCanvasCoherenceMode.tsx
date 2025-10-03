import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectCanvasCoherenceMode, setCanvasCoherenceMode } from 'features/controlLayers/store/paramsSlice';
import { isParameterCanvasCoherenceMode } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasCoherenceMode = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceMode = useAppSelector(selectCanvasCoherenceMode);
  const { t } = useTranslation();

  const options = useMemo<ComboboxOption[]>(
    () => [
      { label: t('parameters.gaussianBlur'), value: 'Gaussian Blur' },
      { label: t('parameters.boxBlur'), value: 'Box Blur' },
      { label: t('parameters.staged'), value: 'Staged' },
    ],
    [t]
  );

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterCanvasCoherenceMode(v?.value)) {
        return;
      }

      dispatch(setCanvasCoherenceMode(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === canvasCoherenceMode), [canvasCoherenceMode, options]);

  return (
    <FormControl>
      <InformationalPopover feature="compositingCoherenceMode">
        <FormLabel>{t('parameters.coherenceMode')}</FormLabel>
      </InformationalPopover>
      <Combobox options={options} value={value} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamCanvasCoherenceMode);

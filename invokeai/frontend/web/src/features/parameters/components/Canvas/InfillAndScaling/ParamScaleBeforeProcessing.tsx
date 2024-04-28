import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { setBoundingBoxScaleMethod } from 'features/canvas/store/canvasSlice';
import { isBoundingBoxScaleMethod } from 'features/canvas/store/canvasTypes';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamScaleBeforeProcessing = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const boundingBoxScaleMethod = useAppSelector((s) => s.canvas.boundingBoxScaleMethod);
  const optimalDimension = useAppSelector(selectOptimalDimension);

  const OPTIONS: ComboboxOption[] = useMemo(
    () => [
      { label: t('modelManager.none'), value: 'none' },
      { label: t('common.auto'), value: 'auto' },
      { label: t('modelManager.manual'), value: 'manual' },
    ],
    [t]
  );

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isBoundingBoxScaleMethod(v?.value)) {
        return;
      }
      dispatch(setBoundingBoxScaleMethod(v.value, optimalDimension));
    },
    [dispatch, optimalDimension]
  );

  const value = useMemo(
    () => OPTIONS.find((o) => o.value === boundingBoxScaleMethod),
    [boundingBoxScaleMethod, OPTIONS]
  );

  return (
    <FormControl>
      <InformationalPopover feature="scaleBeforeProcessing">
        <FormLabel>{t('parameters.scaleBeforeProcessing')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={OPTIONS} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamScaleBeforeProcessing);

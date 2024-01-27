import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { setBoundingBoxScaleMethod } from 'features/canvas/store/canvasSlice';
import { isBoundingBoxScaleMethod } from 'features/canvas/store/canvasTypes';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const OPTIONS: ComboboxOption[] = [
  { label: 'None', value: 'none' },
  { label: 'Auto', value: 'auto' },
  { label: 'Manual', value: 'manual' },
];

const ParamScaleBeforeProcessing = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const boundingBoxScaleMethod = useAppSelector((s) => s.canvas.boundingBoxScaleMethod);
  const optimalDimension = useAppSelector(selectOptimalDimension);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isBoundingBoxScaleMethod(v?.value)) {
        return;
      }
      dispatch(setBoundingBoxScaleMethod(v.value, optimalDimension));
    },
    [dispatch, optimalDimension]
  );

  const value = useMemo(() => OPTIONS.find((o) => o.value === boundingBoxScaleMethod), [boundingBoxScaleMethod]);

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

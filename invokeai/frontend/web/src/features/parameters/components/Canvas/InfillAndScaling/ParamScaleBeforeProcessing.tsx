import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { bboxScaleMethodChanged } from 'features/controlLayers/store/canvasV2Slice';
import { isBoundingBoxScaleMethod } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamScaleBeforeProcessing = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const scaleMethod = useAppSelector((s) => s.canvasV2.bbox.scaleMethod);

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
      dispatch(bboxScaleMethodChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => OPTIONS.find((o) => o.value === scaleMethod), [scaleMethod, OPTIONS]);

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

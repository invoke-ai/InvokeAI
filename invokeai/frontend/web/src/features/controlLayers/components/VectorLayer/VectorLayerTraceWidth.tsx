import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ToolWidthPicker } from 'features/controlLayers/components/Tool/ToolWidthPicker';
import { selectTraceTaperEnds, settingsTraceTaperEndsToggled } from 'features/controlLayers/store/canvasSettingsSlice';
import type { ChangeEventHandler } from 'react';
import { Fragment, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const VectorLayerTraceWidth = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const traceTaperEnds = useAppSelector(selectTraceTaperEnds);
  const label = t('controlLayers.vectorEdit.traceWidth');
  const onTaperEndsChange = useCallback<ChangeEventHandler<HTMLInputElement>>(() => {
    dispatch(settingsTraceTaperEndsToggled());
  }, [dispatch]);

  return (
    <Fragment>
      <FormControl w="min-content" flexShrink={0} gap={2}>
        <FormLabel m={0} mt={1} whiteSpace="nowrap">
          {label}
        </FormLabel>
        <ToolWidthPicker mode="trace" ariaLabel={label} />
      </FormControl>
      <FormControl w="min-content" flexShrink={0} gap={2}>
        <FormLabel m={0} whiteSpace="nowrap">
          {t('controlLayers.vectorEdit.taperEnds')}
        </FormLabel>
        <Checkbox isChecked={traceTaperEnds} onChange={onTaperEndsChange} />
      </FormControl>
    </Fragment>
  );
});

VectorLayerTraceWidth.displayName = 'VectorLayerTraceWidth';

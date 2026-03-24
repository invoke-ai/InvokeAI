import { Checkbox, CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { qwenImageEditShiftChanged, selectQwenImageEditShift } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamQwenImageEditShift = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const shift = useAppSelector(selectQwenImageEditShift);
  const isEnabled = shift !== null;

  const onToggle = useCallback(() => {
    dispatch(qwenImageEditShiftChanged(isEnabled ? null : 3.0));
  }, [dispatch, isEnabled]);

  const onChange = useCallback(
    (value: number) => {
      dispatch(qwenImageEditShiftChanged(value));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <Flex gap={2} alignItems="center">
        <Checkbox isChecked={isEnabled} onChange={onToggle} />
        <FormLabel m={0}>{t('modelManager.qwenImageEditShift')}</FormLabel>
      </Flex>
      {isEnabled && (
        <CompositeNumberInput
          value={shift ?? 3.0}
          onChange={onChange}
          min={0.1}
          max={10}
          step={0.1}
          fineStep={0.01}
          defaultValue={3.0}
        />
      )}
    </FormControl>
  );
});

ParamQwenImageEditShift.displayName = 'ParamQwenImageEditShift';

export default ParamQwenImageEditShift;

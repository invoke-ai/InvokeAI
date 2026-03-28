import { Checkbox, CompositeNumberInput, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { qwenImageShiftChanged, selectQwenImageShift } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamQwenImageShift = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const shift = useAppSelector(selectQwenImageShift);
  const isEnabled = shift !== null;

  const onToggle = useCallback(() => {
    dispatch(qwenImageShiftChanged(isEnabled ? null : 3.0));
  }, [dispatch, isEnabled]);

  const onChange = useCallback(
    (value: number) => {
      dispatch(qwenImageShiftChanged(value));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <Flex gap={2} alignItems="center">
        <Checkbox isChecked={isEnabled} onChange={onToggle} />
        <FormLabel m={0}>{t('modelManager.qwenImageShift')}</FormLabel>
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

ParamQwenImageShift.displayName = 'ParamQwenImageShift';

export default ParamQwenImageShift;

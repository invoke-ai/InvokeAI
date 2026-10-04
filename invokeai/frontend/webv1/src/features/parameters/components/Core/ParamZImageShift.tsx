import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectZImageShift, setZImageShift } from 'features/controlLayers/store/paramsSlice';
import type React from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

const CONSTRAINTS = {
  initial: 3,
  sliderMin: 1,
  sliderMax: 7,
  numberInputMin: 0,
  numberInputMax: 10,
  fineStep: 0.1,
  coarseStep: 0.5,
};

const MARKS = [1, 2, 3, 4, 5, 6, 7];

const ParamZImageShift = () => {
  const { t } = useTranslation();
  const shift = useAppSelector(selectZImageShift);
  const dispatch = useAppDispatch();

  const onChange = useCallback((v: number) => dispatch(setZImageShift(v)), [dispatch]);
  const onReset = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dispatch(setZImageShift(null));
    },
    [dispatch]
  );

  const displayValue = shift ?? CONSTRAINTS.initial;

  return (
    <FormControl>
      <FormLabel>
        {t('parameters.shift')}{' '}
        {shift !== null ? (
          <Text as="span" cursor="pointer" onClick={onReset} display="inline-flex" verticalAlign="middle">
            <PiXBold />
          </Text>
        ) : (
          <Text as="span" opacity={0.5} fontWeight="normal" fontSize="xs">
            ({t('common.auto').toLowerCase()})
          </Text>
        )}
      </FormLabel>
      <CompositeSlider
        value={displayValue}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={displayValue}
        defaultValue={CONSTRAINTS.initial}
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        step={CONSTRAINTS.coarseStep}
        fineStep={CONSTRAINTS.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamZImageShift);

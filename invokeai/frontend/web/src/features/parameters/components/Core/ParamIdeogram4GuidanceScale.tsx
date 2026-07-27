import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectIdeogram4GuidanceScale, setIdeogram4GuidanceScale } from 'features/controlLayers/store/paramsSlice';
import type React from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

// The preset's main per-step guidance weight (gw) is 7.0; shown as the "auto" default.
const PRESET_MAIN_GW = 7;
const MARKS = [1, 4, 7, 10, 12];

// Optional override of the main guidance weight. null = use the preset's guidance schedule.
const ParamIdeogram4GuidanceScale = () => {
  const { t } = useTranslation();
  const guidanceScale = useAppSelector(selectIdeogram4GuidanceScale);
  const dispatch = useAppDispatch();

  const onChange = useCallback((v: number) => dispatch(setIdeogram4GuidanceScale(v)), [dispatch]);
  const onReset = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dispatch(setIdeogram4GuidanceScale(null));
    },
    [dispatch]
  );

  const displayValue = guidanceScale ?? PRESET_MAIN_GW;

  return (
    <FormControl>
      <FormLabel>
        {t('parameters.guidance')}{' '}
        {guidanceScale !== null ? (
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
        defaultValue={PRESET_MAIN_GW}
        min={1}
        max={12}
        step={0.5}
        fineStep={0.1}
        onChange={onChange}
        marks={MARKS}
      />
      <CompositeNumberInput
        value={displayValue}
        defaultValue={PRESET_MAIN_GW}
        min={1}
        max={20}
        step={0.5}
        fineStep={0.1}
        onChange={onChange}
      />
    </FormControl>
  );
};

export default memo(ParamIdeogram4GuidanceScale);

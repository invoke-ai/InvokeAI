import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectHiDiffusionEnabled,
  selectHiDiffusionRauNetEnabled,
  selectHiDiffusionT1Ratio,
  selectHiDiffusionT2Ratio,
  selectHiDiffusionWindowAttnEnabled,
  setHiDiffusionEnabled,
  setHiDiffusionRauNetEnabled,
  setHiDiffusionT1Ratio,
  setHiDiffusionT2Ratio,
  setHiDiffusionWindowAttnEnabled,
} from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamHiDiffusionToggle = memo(() => {
  const hiDiffusionEnabled = useAppSelector(selectHiDiffusionEnabled);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      dispatch(setHiDiffusionEnabled(event.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="hidiffusion">
        <FormLabel maxW="100%" whiteSpace="nowrap" overflow="hidden" textOverflow="ellipsis">
          {t('parameters.hiDiffusion')}
        </FormLabel>
      </InformationalPopover>
      <Switch isChecked={hiDiffusionEnabled} onChange={onChange} />
    </FormControl>
  );
});

ParamHiDiffusionToggle.displayName = 'ParamHiDiffusionToggle';

export const ParamHiDiffusionRauNetToggle = memo(() => {
  const hiDiffusionEnabled = useAppSelector(selectHiDiffusionEnabled);
  const hiDiffusionRauNetEnabled = useAppSelector(selectHiDiffusionRauNetEnabled);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      dispatch(setHiDiffusionRauNetEnabled(event.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="hidiffusionRauNet">
        <FormLabel maxW="100%" whiteSpace="nowrap" overflow="hidden" textOverflow="ellipsis">
          {t('parameters.hiDiffusionRauNet')}
        </FormLabel>
      </InformationalPopover>
      <Switch isChecked={hiDiffusionRauNetEnabled} isDisabled={!hiDiffusionEnabled} onChange={onChange} />
    </FormControl>
  );
});

ParamHiDiffusionRauNetToggle.displayName = 'ParamHiDiffusionRauNetToggle';

export const ParamHiDiffusionWindowAttnToggle = memo(() => {
  const hiDiffusionEnabled = useAppSelector(selectHiDiffusionEnabled);
  const hiDiffusionWindowAttnEnabled = useAppSelector(selectHiDiffusionWindowAttnEnabled);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      dispatch(setHiDiffusionWindowAttnEnabled(event.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="hidiffusionWindowAttn">
        <FormLabel maxW="100%" whiteSpace="nowrap" overflow="hidden" textOverflow="ellipsis">
          {t('parameters.hiDiffusionWindowAttn')}
        </FormLabel>
      </InformationalPopover>
      <Switch isChecked={hiDiffusionWindowAttnEnabled} isDisabled={!hiDiffusionEnabled} onChange={onChange} />
    </FormControl>
  );
});

ParamHiDiffusionWindowAttnToggle.displayName = 'ParamHiDiffusionWindowAttnToggle';

const RATIO_CONSTRAINTS = {
  t1: {
    initial: 0.4,
    sliderMin: 0.1,
    sliderMax: 1,
    numberInputMin: 0.1,
    numberInputMax: 1,
    coarseStep: 0.05,
    fineStep: 0.01,
  },
  t2: {
    initial: 0.0,
    sliderMin: 0.0,
    sliderMax: 1,
    numberInputMin: 0.0,
    numberInputMax: 1,
    coarseStep: 0.05,
    fineStep: 0.01,
  },
} as const;

export const ParamHiDiffusionT1Ratio = memo(() => {
  const hiDiffusionEnabled = useAppSelector(selectHiDiffusionEnabled);
  const hiDiffusionT1Ratio = useAppSelector(selectHiDiffusionT1Ratio);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback((value: number) => dispatch(setHiDiffusionT1Ratio(value)), [dispatch]);

  return (
    <FormControl isDisabled={!hiDiffusionEnabled} gridColumn="1 / -1">
      <InformationalPopover feature="hidiffusionT1Ratio">
        <FormLabel minW="9rem" maxW="100%" whiteSpace="nowrap" overflow="hidden" textOverflow="ellipsis">
          {t('parameters.hiDiffusionT1Ratio')}
        </FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={hiDiffusionT1Ratio}
        defaultValue={RATIO_CONSTRAINTS.t1.initial}
        min={RATIO_CONSTRAINTS.t1.sliderMin}
        max={RATIO_CONSTRAINTS.t1.sliderMax}
        step={RATIO_CONSTRAINTS.t1.coarseStep}
        fineStep={RATIO_CONSTRAINTS.t1.fineStep}
        onChange={onChange}
        marks
      />
      <CompositeNumberInput
        value={hiDiffusionT1Ratio}
        defaultValue={RATIO_CONSTRAINTS.t1.initial}
        min={RATIO_CONSTRAINTS.t1.numberInputMin}
        max={RATIO_CONSTRAINTS.t1.numberInputMax}
        step={RATIO_CONSTRAINTS.t1.coarseStep}
        fineStep={RATIO_CONSTRAINTS.t1.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
});

ParamHiDiffusionT1Ratio.displayName = 'ParamHiDiffusionT1Ratio';

export const ParamHiDiffusionT2Ratio = memo(() => {
  const hiDiffusionEnabled = useAppSelector(selectHiDiffusionEnabled);
  const hiDiffusionT2Ratio = useAppSelector(selectHiDiffusionT2Ratio);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback((value: number) => dispatch(setHiDiffusionT2Ratio(value)), [dispatch]);

  return (
    <FormControl isDisabled={!hiDiffusionEnabled} gridColumn="1 / -1">
      <InformationalPopover feature="hidiffusionT2Ratio">
        <FormLabel minW="9rem" maxW="100%" whiteSpace="nowrap" overflow="hidden" textOverflow="ellipsis">
          {t('parameters.hiDiffusionT2Ratio')}
        </FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={hiDiffusionT2Ratio}
        defaultValue={RATIO_CONSTRAINTS.t2.initial}
        min={RATIO_CONSTRAINTS.t2.sliderMin}
        max={RATIO_CONSTRAINTS.t2.sliderMax}
        step={RATIO_CONSTRAINTS.t2.coarseStep}
        fineStep={RATIO_CONSTRAINTS.t2.fineStep}
        onChange={onChange}
        marks
      />
      <CompositeNumberInput
        value={hiDiffusionT2Ratio}
        defaultValue={RATIO_CONSTRAINTS.t2.initial}
        min={RATIO_CONSTRAINTS.t2.numberInputMin}
        max={RATIO_CONSTRAINTS.t2.numberInputMax}
        step={RATIO_CONSTRAINTS.t2.coarseStep}
        fineStep={RATIO_CONSTRAINTS.t2.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
});

ParamHiDiffusionT2Ratio.displayName = 'ParamHiDiffusionT2Ratio';

import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import {
  Combobox,
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  FormControlGroup,
  FormLabel,
  StandaloneAccordion,
  Switch,
} from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectHrfEnabled,
  selectHrfFinalDimensions,
  selectHrfLatentInterpolationMode,
  selectHrfScale,
  selectHrfStrength,
  selectIsRefinerModelSelected,
  selectModelSupportsHrf,
  setHrfEnabled,
  setHrfLatentInterpolationMode,
  setHrfScale,
  setHrfStrength,
} from 'features/controlLayers/store/paramsSlice';
import { zHrfLatentInterpolationMode } from 'features/controlLayers/store/types';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const SCALE_CONSTRAINTS = {
  initial: 2,
  sliderMin: 1,
  sliderMax: 4,
  numberInputMin: 1,
  numberInputMax: 8,
  coarseStep: 0.05,
  fineStep: 0.01,
};

const STRENGTH_CONSTRAINTS = {
  initial: 0.45,
  sliderMin: 0,
  sliderMax: 1,
  numberInputMin: 0,
  numberInputMax: 1,
  coarseStep: 0.01,
  fineStep: 0.01,
};

const selectBadges = createMemoizedSelector(
  [selectHrfEnabled, selectHrfScale, selectHrfStrength, selectHrfFinalDimensions],
  (enabled, scale, strength, finalDimensions) => {
    if (!enabled) {
      return EMPTY_ARRAY;
    }

    return [`${scale}x`, `${Math.round(strength * 100)}%`, `${finalDimensions.width}x${finalDimensions.height}`];
  }
);

const ParamHrfEnabled = memo(() => {
  const dispatch = useAppDispatch();
  const enabled = useAppSelector(selectHrfEnabled);
  const { t } = useTranslation();

  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      dispatch(setHrfEnabled(event.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl w="min-content">
      <InformationalPopover feature="paramHrf">
        <FormLabel m={0}>{t('hrf.enableHrf')}</FormLabel>
      </InformationalPopover>
      <Switch isChecked={enabled} onChange={onChange} />
    </FormControl>
  );
});

ParamHrfEnabled.displayName = 'ParamHrfEnabled';

const ParamHrfScale = memo(() => {
  const dispatch = useAppDispatch();
  const scale = useAppSelector(selectHrfScale);
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setHrfScale(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="scale">
        <FormLabel>{t('hrf.scale')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={scale}
        defaultValue={SCALE_CONSTRAINTS.initial}
        min={SCALE_CONSTRAINTS.sliderMin}
        max={SCALE_CONSTRAINTS.sliderMax}
        step={SCALE_CONSTRAINTS.coarseStep}
        fineStep={SCALE_CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={[SCALE_CONSTRAINTS.sliderMin, SCALE_CONSTRAINTS.initial, SCALE_CONSTRAINTS.sliderMax]}
      />
      <CompositeNumberInput
        value={scale}
        defaultValue={SCALE_CONSTRAINTS.initial}
        min={SCALE_CONSTRAINTS.numberInputMin}
        max={SCALE_CONSTRAINTS.numberInputMax}
        step={SCALE_CONSTRAINTS.coarseStep}
        fineStep={SCALE_CONSTRAINTS.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
});

ParamHrfScale.displayName = 'ParamHrfScale';

const ParamHrfStrength = memo(() => {
  const dispatch = useAppDispatch();
  const strength = useAppSelector(selectHrfStrength);
  const { t } = useTranslation();

  const onChange = useCallback(
    (v: number) => {
      dispatch(setHrfStrength(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramDenoisingStrength">
        <FormLabel>{t('hrf.strength')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        value={strength}
        defaultValue={STRENGTH_CONSTRAINTS.initial}
        min={STRENGTH_CONSTRAINTS.sliderMin}
        max={STRENGTH_CONSTRAINTS.sliderMax}
        step={STRENGTH_CONSTRAINTS.coarseStep}
        fineStep={STRENGTH_CONSTRAINTS.fineStep}
        onChange={onChange}
        marks={[STRENGTH_CONSTRAINTS.sliderMin, STRENGTH_CONSTRAINTS.initial, STRENGTH_CONSTRAINTS.sliderMax]}
      />
      <CompositeNumberInput
        value={strength}
        defaultValue={STRENGTH_CONSTRAINTS.initial}
        min={STRENGTH_CONSTRAINTS.numberInputMin}
        max={STRENGTH_CONSTRAINTS.numberInputMax}
        step={STRENGTH_CONSTRAINTS.coarseStep}
        fineStep={STRENGTH_CONSTRAINTS.fineStep}
        onChange={onChange}
      />
    </FormControl>
  );
});

ParamHrfStrength.displayName = 'ParamHrfStrength';

const ParamHrfLatentInterpolationMode = memo(() => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectHrfLatentInterpolationMode);
  const { t } = useTranslation();

  const options = useMemo(
    () => [
      { label: t('hrf.bilinear'), value: 'bilinear' },
      { label: t('hrf.bicubic'), value: 'bicubic' },
      { label: t('hrf.nearest'), value: 'nearest' },
      { label: t('hrf.nearestExact'), value: 'nearest-exact' },
      { label: t('hrf.area'), value: 'area' },
    ],
    [t]
  );

  const value = useMemo(() => options.find((o) => o.value === mode), [mode, options]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      const result = zHrfLatentInterpolationMode.safeParse(v?.value);
      if (!result.success) {
        return;
      }
      dispatch(setHrfLatentInterpolationMode(result.data));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramUpscaleMethod">
        <FormLabel>{t('hrf.latentInterpolationMode')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
});

ParamHrfLatentInterpolationMode.displayName = 'ParamHrfLatentInterpolationMode';

export const HighResFixSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const badges = useAppSelector(selectBadges);
  const enabled = useAppSelector(selectHrfEnabled);
  const modelSupportsHrf = useAppSelector(selectModelSupportsHrf);
  const isRefinerModelSelected = useAppSelector(selectIsRefinerModelSelected);
  const { isOpen, onToggle } = useStandaloneAccordionToggle({
    id: 'high-res-fix-settings-generate-tab',
    defaultIsOpen: false,
  });

  if (!modelSupportsHrf || isRefinerModelSelected) {
    return null;
  }

  return (
    <StandaloneAccordion label={t('hrf.hrf')} badges={badges} isOpen={isOpen} onToggle={onToggle}>
      <Flex px={4} pt={4} pb={4} w="full" h="full" flexDir="column" gap={4}>
        <ParamHrfEnabled />
        {enabled && (
          <FormControlGroup>
            <ParamHrfScale />
            <ParamHrfStrength />
            <ParamHrfLatentInterpolationMode />
          </FormControlGroup>
        )}
      </Flex>
    </StandaloneAccordion>
  );
});

HighResFixSettingsAccordion.displayName = 'HighResFixSettingsAccordion';

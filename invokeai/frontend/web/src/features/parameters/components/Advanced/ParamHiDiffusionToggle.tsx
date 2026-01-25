import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  selectHiDiffusionEnabled,
  selectHiDiffusionRauNetEnabled,
  selectHiDiffusionWindowAttnEnabled,
  setHiDiffusionEnabled,
  setHiDiffusionRauNetEnabled,
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

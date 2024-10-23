import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectAutoProcessFilter,
  settingsAutoProcessFilterToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const FilterAutoProcessSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const autoProcessFilter = useAppSelector(selectAutoProcessFilter);

  const onChangeAutoProcessFilter = useCallback(() => {
    dispatch(settingsAutoProcessFilterToggled());
  }, [dispatch]);

  return (
    <FormControl w="min-content">
      <FormLabel m={0}>{t('controlLayers.filter.autoProcess')}</FormLabel>
      <Switch size="sm" isChecked={autoProcessFilter} onChange={onChangeAutoProcessFilter} />
    </FormControl>
  );
});

FilterAutoProcessSwitch.displayName = 'FilterAutoProcessSwitch';

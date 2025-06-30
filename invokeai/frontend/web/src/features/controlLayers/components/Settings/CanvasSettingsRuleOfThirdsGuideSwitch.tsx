import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectRuleOfThirds, settingsRuleOfThirdsToggled } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsRuleOfThirdsSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const ruleOfThirds = useAppSelector(selectRuleOfThirds);
  const onChange = useCallback(() => {
    dispatch(settingsRuleOfThirdsToggled());
  }, [dispatch]);

  return (
    <FormControl>
      <FormLabel m={0} flexGrow={1}>
        {t('controlLayers.ruleOfThirds')}
      </FormLabel>
      <Switch size="sm" isChecked={ruleOfThirds} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsRuleOfThirdsSwitch.displayName = 'CanvasSettingsRuleOfThirdsSwitch';

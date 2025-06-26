import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectRuleOfFourGuide, settingsRuleOfFourGuideToggled } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasSettingsRuleOfFourGuideSwitch = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const ruleOfFourGuide = useAppSelector(selectRuleOfFourGuide);
  const onChange = useCallback(() => {
    dispatch(settingsRuleOfFourGuideToggled());
  }, [dispatch]);

  return (
    <FormControl>
      <FormLabel m={0} flexGrow={1}>
        {t('controlLayers.ruleOfFourGuide')}
      </FormLabel>
      <Switch size="sm" isChecked={ruleOfFourGuide} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsRuleOfFourGuideSwitch.displayName = 'CanvasSettingsRuleOfFourGuideSwitch';
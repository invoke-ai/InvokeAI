import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isEnabledChanged } from 'features/regionalPrompts/store/regionalPromptsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const RPEnabledSwitch = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isEnabled = useAppSelector((s) => s.regionalPrompts.present.isEnabled);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(isEnabledChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl flexGrow={0} gap={2} w="min-content">
      <FormLabel m={0}>Enable RP</FormLabel>
      <Switch isChecked={isEnabled} onChange={onChange} />
    </FormControl>
  );
});

RPEnabledSwitch.displayName = 'RPEnabledSwitch';

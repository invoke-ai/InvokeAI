import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectStagingAreaAutoSwitch,
  settingsStagingAreaAutoSwitchChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { PiCaretLineRightBold, PiCaretRightBold, PiMoonBold } from 'react-icons/pi';

export const StagingAreaAutoSwitchButtons = memo(() => {
  const autoSwitch = useAppSelector(selectStagingAreaAutoSwitch);
  const dispatch = useAppDispatch();

  const onClickOff = useCallback(() => {
    dispatch(settingsStagingAreaAutoSwitchChanged('off'));
  }, [dispatch]);
  const onClickSwitchOnStart = useCallback(() => {
    dispatch(settingsStagingAreaAutoSwitchChanged('switch_on_start'));
  }, [dispatch]);
  const onClickSwitchOnFinished = useCallback(() => {
    dispatch(settingsStagingAreaAutoSwitchChanged('switch_on_finish'));
  }, [dispatch]);

  return (
    <>
      <IconButton
        aria-label="Do not auto-switch"
        tooltip="Do not auto-switch"
        icon={<PiMoonBold />}
        colorScheme={autoSwitch === 'off' ? 'invokeBlue' : 'base'}
        onClick={onClickOff}
      />
      <IconButton
        aria-label="Switch on start"
        tooltip="Switch on start"
        icon={<PiCaretRightBold />}
        colorScheme={autoSwitch === 'switch_on_start' ? 'invokeBlue' : 'base'}
        onClick={onClickSwitchOnStart}
      />
      <IconButton
        aria-label="Switch on finish"
        tooltip="Switch on finish"
        icon={<PiCaretLineRightBold />}
        colorScheme={autoSwitch === 'switch_on_finish' ? 'invokeBlue' : 'base'}
        onClick={onClickSwitchOnFinished}
      />
    </>
  );
});
StagingAreaAutoSwitchButtons.displayName = 'StagingAreaAutoSwitchButtons';

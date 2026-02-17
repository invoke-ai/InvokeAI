import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAITooltip } from 'common/components/IAITooltip';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import {
  selectStagingAreaAutoSwitch,
  settingsStagingAreaAutoSwitchChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { PiCaretLineRightBold, PiCaretRightBold, PiMoonBold } from 'react-icons/pi';

export const StagingAreaAutoSwitchButtons = memo(() => {
  const canvasManager = useCanvasManager();
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);

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
      <IAITooltip label="Do not auto-switch">
        <IconButton
          aria-label="Do not auto-switch"
          icon={<PiMoonBold />}
          colorScheme={autoSwitch === 'off' ? 'invokeBlue' : 'base'}
          onClick={onClickOff}
          isDisabled={!shouldShowStagedImage}
        />
      </IAITooltip>
      <IAITooltip label="Switch on start">
        <IconButton
          aria-label="Switch on start"
          icon={<PiCaretRightBold />}
          colorScheme={autoSwitch === 'switch_on_start' ? 'invokeBlue' : 'base'}
          onClick={onClickSwitchOnStart}
          isDisabled={!shouldShowStagedImage}
        />
      </IAITooltip>
      <IAITooltip label="Switch on finish">
        <IconButton
          aria-label="Switch on finish"
          icon={<PiCaretLineRightBold />}
          colorScheme={autoSwitch === 'switch_on_finish' ? 'invokeBlue' : 'base'}
          onClick={onClickSwitchOnFinished}
          isDisabled={!shouldShowStagedImage}
        />
      </IAITooltip>
    </>
  );
});
StagingAreaAutoSwitchButtons.displayName = 'StagingAreaAutoSwitchButtons';

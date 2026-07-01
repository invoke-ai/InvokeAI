import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import {
  selectStagingAreaAutoSwitch,
  settingsStagingAreaAutoSwitchChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretLineRightBold, PiCaretRightBold, PiMoonBold } from 'react-icons/pi';

export const StagingAreaAutoSwitchButtons = memo(() => {
  const { t } = useTranslation();
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
      <IconButton
        aria-label={t('controlLayers.autoSwitch.doNotAutoSwitch')}
        tooltip={t('controlLayers.autoSwitch.doNotAutoSwitch')}
        icon={<PiMoonBold />}
        colorScheme={autoSwitch === 'off' ? 'invokeBlue' : 'base'}
        onClick={onClickOff}
        isDisabled={!shouldShowStagedImage}
      />
      <IconButton
        aria-label={t('controlLayers.autoSwitch.switchOnStartDesc')}
        tooltip={t('controlLayers.autoSwitch.switchOnStartDesc')}
        icon={<PiCaretRightBold />}
        colorScheme={autoSwitch === 'switch_on_start' ? 'invokeBlue' : 'base'}
        onClick={onClickSwitchOnStart}
        isDisabled={!shouldShowStagedImage}
      />
      <IconButton
        aria-label={t('controlLayers.autoSwitch.switchOnFinishDesc')}
        tooltip={t('controlLayers.autoSwitch.switchOnFinishDesc')}
        icon={<PiCaretLineRightBold />}
        colorScheme={autoSwitch === 'switch_on_finish' ? 'invokeBlue' : 'base'}
        onClick={onClickSwitchOnFinished}
        isDisabled={!shouldShowStagedImage}
      />
    </>
  );
});
StagingAreaAutoSwitchButtons.displayName = 'StagingAreaAutoSwitchButtons';

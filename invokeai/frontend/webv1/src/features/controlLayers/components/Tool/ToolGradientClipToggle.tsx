import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectGradientClipEnabled,
  settingsGradientClipToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCropBold } from 'react-icons/pi';

import { TOOL_OPTION_ICON_SIZE } from './toolOptionIconSize';

export const ToolGradientClipToggle = memo(() => {
  const { t } = useTranslation();
  const isEnabled = useAppSelector(selectGradientClipEnabled);
  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    dispatch(settingsGradientClipToggled());
  }, [dispatch]);

  const label = t('controlLayers.gradient.clip');

  return (
    <Tooltip label={label}>
      <IconButton
        aria-label={label}
        icon={<PiCropBold size={TOOL_OPTION_ICON_SIZE} />}
        size="sm"
        variant="solid"
        colorScheme={isEnabled ? 'invokeBlue' : 'base'}
        onClick={onClick}
      />
    </Tooltip>
  );
});

ToolGradientClipToggle.displayName = 'ToolGradientClipToggle';

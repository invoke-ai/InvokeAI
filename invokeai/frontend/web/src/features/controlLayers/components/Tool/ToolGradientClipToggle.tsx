import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAITooltip } from 'common/components/IAITooltip';
import {
  selectGradientClipEnabled,
  settingsGradientClipToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCropBold } from 'react-icons/pi';

export const ToolGradientClipToggle = memo(() => {
  const { t } = useTranslation();
  const isEnabled = useAppSelector(selectGradientClipEnabled);
  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    dispatch(settingsGradientClipToggled());
  }, [dispatch]);

  const label = t('controlLayers.gradient.clip', { defaultValue: 'Clip Gradient' });

  return (
    <IAITooltip label={label}>
      <IconButton
        aria-label={label}
        icon={<PiCropBold size={16} />}
        size="sm"
        variant="solid"
        colorScheme={isEnabled ? 'invokeBlue' : 'base'}
        onClick={onClick}
      />
    </IAITooltip>
  );
});

ToolGradientClipToggle.displayName = 'ToolGradientClipToggle';

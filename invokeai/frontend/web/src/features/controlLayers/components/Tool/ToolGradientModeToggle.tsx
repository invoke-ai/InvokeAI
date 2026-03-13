import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAITooltip } from 'common/components/IAITooltip';
import { GradientLinearIcon, GradientRadialIcon } from 'features/controlLayers/components/Tool/GradientIcons';
import { selectGradientType, settingsGradientTypeChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ToolGradientModeToggle = memo(() => {
  const { t } = useTranslation();
  const gradientType = useAppSelector(selectGradientType);
  const dispatch = useAppDispatch();

  const onLinearClick = useCallback(() => dispatch(settingsGradientTypeChanged('linear')), [dispatch]);
  const onRadialClick = useCallback(() => dispatch(settingsGradientTypeChanged('radial')), [dispatch]);

  return (
    <ButtonGroup isAttached size="sm">
      <IAITooltip label={t('controlLayers.gradient.linear', { defaultValue: 'Linear' })}>
        <IconButton
          aria-label={t('controlLayers.gradient.linear', { defaultValue: 'Linear' })}
          icon={<GradientLinearIcon />}
          colorScheme={gradientType === 'linear' ? 'invokeBlue' : 'base'}
          variant="solid"
          onClick={onLinearClick}
        />
      </IAITooltip>
      <IAITooltip label={t('controlLayers.gradient.radial', { defaultValue: 'Radial' })}>
        <IconButton
          aria-label={t('controlLayers.gradient.radial', { defaultValue: 'Radial' })}
          icon={<GradientRadialIcon />}
          colorScheme={gradientType === 'radial' ? 'invokeBlue' : 'base'}
          variant="solid"
          onClick={onRadialClick}
        />
      </IAITooltip>
    </ButtonGroup>
  );
});

ToolGradientModeToggle.displayName = 'ToolGradientModeToggle';

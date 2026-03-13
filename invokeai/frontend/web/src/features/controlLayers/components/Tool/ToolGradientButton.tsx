import { IconButton } from '@invoke-ai/ui-library';
import { IAITooltip } from 'common/components/IAITooltip';
import { GradientToolIcon } from 'features/controlLayers/components/Tool/GradientIcons';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ToolGradientButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('gradient');
  const selectGradient = useSelectTool('gradient');
  // clicking selects the gradient tool; mode switching is handled in the top toolbar
  const handleClick = useCallback(() => selectGradient(), [selectGradient]);

  const gradientLabel = t('controlLayers.tool.gradient', { defaultValue: 'Gradient' });

  return (
    <IAITooltip label={gradientLabel} placement="end">
      <IconButton
        aria-label={gradientLabel}
        icon={<GradientToolIcon />}
        isActive={isSelected}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={handleClick}
      />
    </IAITooltip>
  );
});

ToolGradientButton.displayName = 'ToolGradientButton';

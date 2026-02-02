import { IconButton, Tooltip } from '@invoke-ai/ui-library';
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
    <Tooltip label={gradientLabel} placement="end">
      <IconButton
        aria-label={gradientLabel}
        icon={<GradientToolIcon />}
        isActive={isSelected}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={handleClick}
      />
    </Tooltip>
  );
});

ToolGradientButton.displayName = 'ToolGradientButton';

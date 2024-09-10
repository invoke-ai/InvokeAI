import { IconButton } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { memo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiPaintBrushBold } from 'react-icons/pi';

export const ToolBrushButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('brush');
  const selectBrush = useSelectTool('brush');

  useHotkeys('b', selectBrush, { enabled: !isSelected }, [isSelected, selectBrush]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.brush')} (B)`}
      tooltip={`${t('controlLayers.tool.brush')} (B)`}
      icon={<PiPaintBrushBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="solid"
      onClick={selectBrush}
      isDisabled={isSelected}
    />
  );
});

ToolBrushButton.displayName = 'ToolBrushButton';

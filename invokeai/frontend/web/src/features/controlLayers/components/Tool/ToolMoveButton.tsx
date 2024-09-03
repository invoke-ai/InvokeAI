import { IconButton } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { memo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiCursorBold } from 'react-icons/pi';

export const ToolMoveButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('move');
  const selectMove = useSelectTool('move');

  useHotkeys('v', selectMove, { enabled: !isSelected }, [isSelected, selectMove]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.move')} (V)`}
      tooltip={`${t('controlLayers.tool.move')} (V)`}
      icon={<PiCursorBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="solid"
      onClick={selectMove}
      isDisabled={isSelected}
    />
  );
});

ToolMoveButton.displayName = 'ToolMoveButton';

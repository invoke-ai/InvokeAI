import { IconButton } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { memo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiRectangleBold } from 'react-icons/pi';

export const ToolRectButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('rect');
  const selectRect = useSelectTool('rect');

  useHotkeys('u', selectRect, { enabled: !isSelected }, [isSelected, selectRect]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.rectangle')} (U)`}
      tooltip={`${t('controlLayers.tool.rectangle')} (U)`}
      icon={<PiRectangleBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="solid"
      onClick={selectRect}
      isDisabled={isSelected}
    />
  );
});

ToolRectButton.displayName = 'ToolRectButton';

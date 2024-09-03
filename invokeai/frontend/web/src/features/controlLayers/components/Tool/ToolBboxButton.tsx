import { IconButton } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { memo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold } from 'react-icons/pi';

export const ToolBboxButton = memo(() => {
  const { t } = useTranslation();
  const selectBbox = useSelectTool('bbox');
  const isSelected = useToolIsSelected('bbox');

  useHotkeys('c', selectBbox, { enabled: !isSelected }, [selectBbox, isSelected]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.bbox')} (C)`}
      tooltip={`${t('controlLayers.tool.bbox')} (C)`}
      icon={<PiBoundingBoxBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="solid"
      onClick={selectBbox}
      isDisabled={isSelected}
    />
  );
});

ToolBboxButton.displayName = 'ToolBboxButton';

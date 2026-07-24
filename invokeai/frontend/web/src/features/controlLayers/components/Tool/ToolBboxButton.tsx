import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold } from 'react-icons/pi';

export const ToolBboxButton = memo(() => {
  const { t } = useTranslation();
  const selectBbox = useSelectTool('bbox');
  const isSelected = useToolIsSelected('bbox');

  return (
    <Tooltip label={`${t('controlLayers.tool.bbox')} (C)`} placement="end">
      <IconButton
        aria-label={`${t('controlLayers.tool.bbox')} (C)`}
        icon={<PiBoundingBoxBold />}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={selectBbox}
      />
    </Tooltip>
  );
});

ToolBboxButton.displayName = 'ToolBboxButton';

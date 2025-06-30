import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCursorBold } from 'react-icons/pi';

export const ToolMoveButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('move');
  const selectMove = useSelectTool('move');

  useRegisteredHotkeys({
    id: 'selectMoveTool',
    category: 'canvas',
    callback: selectMove,
    options: { enabled: !isSelected },
    dependencies: [isSelected, selectMove],
  });

  return (
    <Tooltip label={`${t('controlLayers.tool.move')} (V)`} placement="end">
      <IconButton
        aria-label={`${t('controlLayers.tool.move')} (V)`}
        icon={<PiCursorBold />}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={selectMove}
      />
    </Tooltip>
  );
});

ToolMoveButton.displayName = 'ToolMoveButton';

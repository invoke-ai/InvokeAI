import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPaintBrushBold } from 'react-icons/pi';

export const ToolBrushButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('brush');
  const selectBrush = useSelectTool('brush');

  useRegisteredHotkeys({
    id: 'selectBrushTool',
    category: 'canvas',
    callback: selectBrush,
    options: { enabled: !isSelected },
    dependencies: [isSelected, selectBrush],
  });

  return (
    <Tooltip label={`${t('controlLayers.tool.brush')} (B)`} placement="end">
      <IconButton
        aria-label={`${t('controlLayers.tool.brush')} (B)`}
        icon={<PiPaintBrushBold />}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={selectBrush}
      />
    </Tooltip>
  );
});

ToolBrushButton.displayName = 'ToolBrushButton';

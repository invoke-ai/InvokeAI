import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShapesBold } from 'react-icons/pi';

export const ToolShapesButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('rect');
  const selectShapes = useSelectTool('rect');
  const label = t('controlLayers.tool.shapes', { defaultValue: 'Shapes' });

  useRegisteredHotkeys({
    id: 'selectRectTool',
    category: 'canvas',
    callback: selectShapes,
    options: { enabled: !isSelected },
    dependencies: [isSelected, selectShapes],
  });

  return (
    <Tooltip label={`${label} (U)`} placement="end">
      <IconButton
        aria-label={`${label} (U)`}
        icon={<PiShapesBold />}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={selectShapes}
      />
    </Tooltip>
  );
});

ToolShapesButton.displayName = 'ToolShapesButton';

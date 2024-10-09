import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEraserBold } from 'react-icons/pi';

export const ToolEraserButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('eraser');
  const selectEraser = useSelectTool('eraser');
  const imageViewer = useImageViewer();

  useRegisteredHotkeys({
    id: 'selectEraserTool',
    category: 'canvas',
    callback: selectEraser,
    options: { enabled: !isSelected && !imageViewer.isOpen },
    dependencies: [isSelected, selectEraser, imageViewer.isOpen],
  });

  return (
    <Tooltip label={`${t('controlLayers.tool.eraser')} (E)`} placement="end">
      <IconButton
        aria-label={`${t('controlLayers.tool.eraser')} (E)`}
        icon={<PiEraserBold />}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={selectEraser}
      />
    </Tooltip>
  );
});

ToolEraserButton.displayName = 'ToolEraserButton';

import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiRectangleBold } from 'react-icons/pi';

export const ToolRectButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('rect');
  const selectRect = useSelectTool('rect');
  const imageViewer = useImageViewer();

  useRegisteredHotkeys({
    id: 'selectRectTool',
    category: 'canvas',
    callback: selectRect,
    options: { enabled: !isSelected && !imageViewer.isOpen },
    dependencies: [isSelected, selectRect, imageViewer.isOpen],
  });

  return (
    <Tooltip label={`${t('controlLayers.tool.rectangle')} (U)`} placement="end">
      <IconButton
        aria-label={`${t('controlLayers.tool.rectangle')} (U)`}
        icon={<PiRectangleBold />}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={selectRect}
      />
    </Tooltip>
  );
});

ToolRectButton.displayName = 'ToolRectButton';

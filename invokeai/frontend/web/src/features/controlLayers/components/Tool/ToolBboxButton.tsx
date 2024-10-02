import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold } from 'react-icons/pi';

export const ToolBboxButton = memo(() => {
  const { t } = useTranslation();
  const selectBbox = useSelectTool('bbox');
  const isSelected = useToolIsSelected('bbox');
  const imageViewer = useImageViewer();

  useRegisteredHotkeys({
    id: 'selectBboxTool',
    category: 'canvas',
    callback: selectBbox,
    options: { enabled: !isSelected && !imageViewer.isOpen },
    dependencies: [selectBbox, isSelected, imageViewer.isOpen],
  });

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

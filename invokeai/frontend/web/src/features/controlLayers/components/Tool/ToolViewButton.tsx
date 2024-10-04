import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiHandBold } from 'react-icons/pi';

export const ToolViewButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('view');
  const selectView = useSelectTool('view');
  const imageViewer = useImageViewer();

  useRegisteredHotkeys({
    id: 'selectViewTool',
    category: 'canvas',
    callback: selectView,
    options: { enabled: !isSelected && !imageViewer.isOpen },
    dependencies: [selectView, isSelected, imageViewer.isOpen],
  });

  return (
    <Tooltip label={`${t('controlLayers.tool.view')} (H)`} placement="end">
      <IconButton
        aria-label={`${t('controlLayers.tool.view')} (H)`}
        icon={<PiHandBold />}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={selectView}
      />
    </Tooltip>
  );
});

ToolViewButton.displayName = 'ToolViewButton';

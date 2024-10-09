import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyedropperBold } from 'react-icons/pi';

export const ToolColorPickerButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('colorPicker');
  const selectColorPicker = useSelectTool('colorPicker');
  const imageViewer = useImageViewer();

  useRegisteredHotkeys({
    id: 'selectColorPickerTool',
    category: 'canvas',
    callback: selectColorPicker,
    options: { enabled: !isSelected && !imageViewer.isOpen },
    dependencies: [selectColorPicker, isSelected, imageViewer.isOpen],
  });

  return (
    <Tooltip label={`${t('controlLayers.tool.colorPicker')} (I)`} placement="end">
      <IconButton
        aria-label={`${t('controlLayers.tool.colorPicker')} (I)`}
        icon={<PiEyedropperBold />}
        colorScheme={isSelected ? 'invokeBlue' : 'base'}
        variant="solid"
        onClick={selectColorPicker}
      />
    </Tooltip>
  );
});

ToolColorPickerButton.displayName = 'ToolColorPickerButton';

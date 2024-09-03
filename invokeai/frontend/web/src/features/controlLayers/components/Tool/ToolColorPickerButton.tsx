import { IconButton } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { memo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiEyedropperBold } from 'react-icons/pi';

export const ToolColorPickerButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('colorPicker');
  const selectColorPicker = useSelectTool('colorPicker');

  useHotkeys('i', selectColorPicker, { enabled: !isSelected }, [selectColorPicker, isSelected]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.colorPicker')} (I)`}
      tooltip={`${t('controlLayers.tool.colorPicker')} (I)`}
      icon={<PiEyedropperBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="solid"
      onClick={selectColorPicker}
      isDisabled={isSelected}
    />
  );
});

ToolColorPickerButton.displayName = 'ToolColorPickerButton';

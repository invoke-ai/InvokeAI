import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useSharedInfoHotkeyTarget } from 'features/system/components/HotkeysModal/useSharedInfoHotkeyTarget';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyedropperBold } from 'react-icons/pi';

export const ToolColorPickerButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('colorPicker');
  const selectColorPicker = useSelectTool('colorPicker');
  const hotkeyTarget = useSharedInfoHotkeyTarget();
  const hotkeyOptions = useMemo(() => ({ enabled: hotkeyTarget === 'colorPicker' }), [hotkeyTarget]);

  useRegisteredHotkeys({
    id: 'selectColorPickerTool',
    category: 'canvas',
    callback: selectColorPicker,
    options: hotkeyOptions,
    dependencies: [selectColorPicker, hotkeyTarget],
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

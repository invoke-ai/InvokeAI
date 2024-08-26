import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useIsFiltering } from 'features/controlLayers/hooks/useIsFiltering';
import { useIsTransforming } from 'features/controlLayers/hooks/useIsTransforming';
import { memo, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiEyedropperBold } from 'react-icons/pi';

export const ToolColorPickerButton = memo(() => {
  const { t } = useTranslation();
  const isFiltering = useIsFiltering();
  const isTransforming = useIsTransforming();
  const selectColorPicker = useSelectTool('colorPicker');
  const isSelected = useToolIsSelected('colorPicker');
  const isStaging = useAppSelector((s) => s.canvasV2.session.isStaging);

  const isDisabled = useMemo(() => {
    return isTransforming || isFiltering || isStaging;
  }, [isFiltering, isStaging, isTransforming]);

  useHotkeys('i', selectColorPicker, { enabled: !isDisabled || isSelected }, [
    selectColorPicker,
    isSelected,
    isDisabled,
  ]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.colorPicker')} (I)`}
      tooltip={`${t('controlLayers.tool.colorPicker')} (I)`}
      icon={<PiEyedropperBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={selectColorPicker}
      isDisabled={isDisabled}
    />
  );
});

ToolColorPickerButton.displayName = 'ToolColorPickerButton';

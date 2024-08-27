import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useIsFiltering } from 'features/controlLayers/hooks/useIsFiltering';
import { useIsTransforming } from 'features/controlLayers/hooks/useIsTransforming';
import { selectIsStaging } from 'features/controlLayers/store/canvasSessionSlice';
import { selectIsSelectedEntityDrawable } from 'features/controlLayers/store/selectors';
import { memo, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiRectangleBold } from 'react-icons/pi';

export const ToolRectButton = memo(() => {
  const { t } = useTranslation();
  const selectRect = useSelectTool('rect');
  const isSelected = useToolIsSelected('rect');
  const isFiltering = useIsFiltering();
  const isTransforming = useIsTransforming();
  const isStaging = useAppSelector(selectIsStaging);
  const isDrawingToolAllowed = useAppSelector(selectIsSelectedEntityDrawable);

  const isDisabled = useMemo(() => {
    return isTransforming || isFiltering || isStaging || !isDrawingToolAllowed;
  }, [isDrawingToolAllowed, isFiltering, isStaging, isTransforming]);

  useHotkeys('u', selectRect, { enabled: !isDisabled || isSelected }, [isDisabled, isSelected, selectRect]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.rectangle')} (U)`}
      tooltip={`${t('controlLayers.tool.rectangle')} (U)`}
      icon={<PiRectangleBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={selectRect}
      isDisabled={isDisabled}
    />
  );
});

ToolRectButton.displayName = 'ToolRectButton';

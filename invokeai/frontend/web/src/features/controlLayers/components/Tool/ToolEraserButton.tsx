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
import { PiEraserBold } from 'react-icons/pi';

export const ToolEraserButton = memo(() => {
  const { t } = useTranslation();
  const isFiltering = useIsFiltering();
  const isTransforming = useIsTransforming();
  const isStaging = useAppSelector(selectIsStaging);
  const selectEraser = useSelectTool('eraser');
  const isSelected = useToolIsSelected('eraser');
  const isDrawingToolAllowed = useAppSelector(selectIsSelectedEntityDrawable);
  const isDisabled = useMemo(() => {
    return isTransforming || isFiltering || isStaging || !isDrawingToolAllowed;
  }, [isDrawingToolAllowed, isFiltering, isStaging, isTransforming]);

  useHotkeys('e', selectEraser, { enabled: !isDisabled || isSelected }, [isDisabled, isSelected, selectEraser]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.eraser')} (E)`}
      tooltip={`${t('controlLayers.tool.eraser')} (E)`}
      icon={<PiEraserBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={selectEraser}
      isDisabled={isDisabled}
    />
  );
});

ToolEraserButton.displayName = 'ToolEraserButton';

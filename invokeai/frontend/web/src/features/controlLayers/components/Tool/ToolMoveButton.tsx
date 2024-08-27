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
import { PiCursorBold } from 'react-icons/pi';

export const ToolMoveButton = memo(() => {
  const { t } = useTranslation();
  const isFiltering = useIsFiltering();
  const isTransforming = useIsTransforming();
  const selectMove = useSelectTool('move');
  const isSelected = useToolIsSelected('move');
  const isStaging = useAppSelector(selectIsStaging);
  const isDrawingToolAllowed = useAppSelector(selectIsSelectedEntityDrawable);
  const isDisabled = useMemo(() => {
    return isTransforming || isFiltering || isStaging || !isDrawingToolAllowed;
  }, [isDrawingToolAllowed, isFiltering, isStaging, isTransforming]);

  useHotkeys('v', selectMove, { enabled: !isDisabled || isSelected }, [isDisabled, isSelected, selectMove]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.move')} (V)`}
      tooltip={`${t('controlLayers.tool.move')} (V)`}
      icon={<PiCursorBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={selectMove}
      isDisabled={isDisabled}
    />
  );
});

ToolMoveButton.displayName = 'ToolMoveButton';

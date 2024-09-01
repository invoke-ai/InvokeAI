import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useIsFiltering } from 'features/controlLayers/hooks/useIsFiltering';
import { useIsTransforming } from 'features/controlLayers/hooks/useIsTransforming';
import { selectIsStaging } from 'features/controlLayers/store/canvasSessionSlice';
import { memo, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold } from 'react-icons/pi';

export const ToolBboxButton = memo(() => {
  const { t } = useTranslation();
  const selectBbox = useSelectTool('bbox');
  const isSelected = useToolIsSelected('bbox');
  const isFiltering = useIsFiltering();
  const isTransforming = useIsTransforming();
  const isStaging = useAppSelector(selectIsStaging);
  const isDisabled = useMemo(() => {
    return isTransforming || isFiltering || isStaging;
  }, [isFiltering, isStaging, isTransforming]);

  useHotkeys('c', selectBbox, { enabled: !isDisabled || isSelected }, [selectBbox, isSelected, isDisabled]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.bbox')} (C)`}
      tooltip={`${t('controlLayers.tool.bbox')} (C)`}
      icon={<PiBoundingBoxBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={selectBbox}
      isDisabled={isDisabled}
    />
  );
});

ToolBboxButton.displayName = 'ToolBboxButton';

import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useIsFiltering } from 'features/controlLayers/hooks/useIsFiltering';
import { useIsTransforming } from 'features/controlLayers/hooks/useIsTransforming';
import { memo, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiHandBold } from 'react-icons/pi';

export const ToolViewButton = memo(() => {
  const { t } = useTranslation();
  const isTransforming = useIsTransforming();
  const isFiltering = useIsFiltering();
  const isStaging = useAppSelector((s) => s.canvasSession.isStaging);
  const selectView = useSelectTool('view');
  const isSelected = useToolIsSelected('view');
  const isDisabled = useMemo(() => {
    return isTransforming || isFiltering || isStaging;
  }, [isFiltering, isStaging, isTransforming]);

  useHotkeys('h', selectView, { enabled: !isDisabled || isSelected }, [selectView, isSelected, isDisabled]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.view')} (H)`}
      tooltip={`${t('controlLayers.tool.view')} (H)`}
      icon={<PiHandBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={selectView}
      isDisabled={isDisabled}
    />
  );
});

ToolViewButton.displayName = 'ToolViewButton';

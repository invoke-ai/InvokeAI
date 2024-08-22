import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsTransforming } from 'features/controlLayers/hooks/useIsTransforming';
import { toolChanged } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiEyedropperBold } from 'react-icons/pi';

export const ToolEyeDropperButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isTransforming = useIsTransforming();
  const isSelected = useAppSelector((s) => s.canvasV2.tool.selected === 'eyeDropper');
  const isStaging = useAppSelector((s) => s.canvasV2.session.isStaging);

  const isDisabled = useMemo(() => {
    return isTransforming || isStaging;
  }, [isStaging, isTransforming]);

  const onClick = useCallback(() => {
    dispatch(toolChanged('eyeDropper'));
  }, [dispatch]);

  useHotkeys('i', onClick, { enabled: !isDisabled || isSelected }, [onClick, isSelected, isDisabled]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.eyeDropper')} (I)`}
      tooltip={`${t('controlLayers.tool.eyeDropper')} (I)`}
      icon={<PiEyedropperBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={onClick}
      isDisabled={isDisabled}
    />
  );
});

ToolEyeDropperButton.displayName = 'ToolEyeDropperButton';

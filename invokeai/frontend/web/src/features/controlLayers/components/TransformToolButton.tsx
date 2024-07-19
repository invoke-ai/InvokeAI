import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { toolChanged } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiResizeBold } from 'react-icons/pi';

export const TransformToolButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector((s) => s.canvasV2.tool.selected === 'transform');
  const isDisabled = useAppSelector(
    (s) => s.canvasV2.selectedEntityIdentifier === null || s.canvasV2.session.isStaging
  );

  const onClick = useCallback(() => {
    dispatch(toolChanged('transform'));
  }, [dispatch]);

  useHotkeys(['ctrl+t', 'meta+t'], onClick, { enabled: !isDisabled }, [isDisabled, onClick]);

  return (
    <IconButton
      aria-label={`${t('unifiedCanvas.transform')} (Ctrl+T)`}
      tooltip={`${t('unifiedCanvas.transform')} (Ctrl+T)`}
      icon={<PiResizeBold />}
      variant={isSelected ? 'solid' : 'outline'}
      onClick={onClick}
      isDisabled={isDisabled}
    />
  );
});

TransformToolButton.displayName = 'TransformToolButton';

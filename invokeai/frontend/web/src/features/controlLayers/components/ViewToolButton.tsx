import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { toolChanged } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiHandBold } from 'react-icons/pi';

export const ViewToolButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector((s) => s.canvasV2.tool.selected === 'view');
  const isDisabled = useAppSelector((s) => s.canvasV2.session.isStaging);
  const onClick = useCallback(() => {
    dispatch(toolChanged('view'));
  }, [dispatch]);

  useHotkeys('h', onClick, [onClick]);

  return (
    <IconButton
      aria-label={`${t('unifiedCanvas.view')} (H)`}
      tooltip={`${t('unifiedCanvas.view')} (H)`}
      icon={<PiHandBold />}
      variant={isSelected ? 'solid' : 'outline'}
      onClick={onClick}
      isDisabled={isDisabled}
    />
  );
});

ViewToolButton.displayName = 'ViewToolButton';

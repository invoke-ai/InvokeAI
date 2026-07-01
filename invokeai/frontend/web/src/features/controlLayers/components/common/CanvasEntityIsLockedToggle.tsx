import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIsLocked } from 'features/controlLayers/hooks/useEntityIsLocked';
import { entityIsLockedToggled } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLockSimpleFill, PiLockSimpleOpenBold } from 'react-icons/pi';

export const CanvasEntityIsLockedToggle = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const isLocked = useEntityIsLocked(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(entityIsLockedToggled({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <IconButton
      size="sm"
      aria-label={t(isLocked ? 'controlLayers.locked' : 'controlLayers.unlocked')}
      tooltip={t(isLocked ? 'controlLayers.locked' : 'controlLayers.unlocked')}
      variant="link"
      alignSelf="stretch"
      icon={isLocked ? <PiLockSimpleFill /> : <PiLockSimpleOpenBold />}
      onClick={onClick}
      isDisabled={isBusy}
    />
  );
});

CanvasEntityIsLockedToggle.displayName = 'CanvasEntityIsLockedToggle';

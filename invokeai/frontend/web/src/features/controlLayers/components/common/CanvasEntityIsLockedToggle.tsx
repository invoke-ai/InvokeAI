import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useBoolean } from 'common/hooks/useBoolean';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityIsLocked } from 'features/controlLayers/hooks/useEntityIsLocked';
import { entityIsLockedToggled } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLockSimpleFill } from 'react-icons/pi';

export const CanvasEntityIsLockedToggle = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const ref = useRef<HTMLButtonElement>(null);
  const isLocked = useEntityIsLocked(entityIdentifier);
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(entityIsLockedToggled({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);
  const isHovered = useBoolean(false);

  return (
    <IconButton
      ref={ref}
      size="sm"
      onMouseOver={isHovered.setTrue}
      onMouseOut={isHovered.setFalse}
      aria-label={t(isLocked ? 'controlLayers.locked' : 'controlLayers.unlocked')}
      tooltip={t(isLocked ? 'controlLayers.locked' : 'controlLayers.unlocked')}
      variant="ghost"
      icon={isLocked || isHovered.isTrue ? <PiLockSimpleFill /> : undefined}
      onClick={onClick}
    />
  );
});

CanvasEntityIsLockedToggle.displayName = 'CanvasEntityIsLockedToggle';

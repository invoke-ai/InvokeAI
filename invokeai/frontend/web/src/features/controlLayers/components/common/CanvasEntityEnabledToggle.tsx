import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { stopPropagation } from 'common/util/stopPropagation';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityIsEnabled } from 'features/controlLayers/hooks/useEntityIsEnabled';
import { entityIsEnabledToggled } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';

export const CanvasEntityEnabledToggle = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const isEnabled = useEntityIsEnabled(entityIdentifier);
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(entityIsEnabledToggled({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <IconButton
      size="sm"
      aria-label={t(isEnabled ? 'common.enabled' : 'common.disabled')}
      tooltip={t(isEnabled ? 'common.enabled' : 'common.disabled')}
      variant="outline"
      icon={isEnabled ? <PiCheckBold /> : undefined}
      onClick={onClick}
      colorScheme="base"
      onDoubleClick={stopPropagation} // double click expands the layer
    />
  );
});

CanvasEntityEnabledToggle.displayName = 'CanvasEntityEnabledToggle';

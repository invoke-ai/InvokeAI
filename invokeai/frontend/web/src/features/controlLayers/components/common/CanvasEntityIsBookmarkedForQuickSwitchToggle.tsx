import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityIsBookmarkedForQuickSwitch } from 'features/controlLayers/hooks/useEntityIsBookmarkedForQuickSwitch';
import { bookmarkedEntityChanged } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBookmarkSimpleBold, PiBookmarkSimpleFill } from 'react-icons/pi';

export const CanvasEntityIsBookmarkedForQuickSwitchToggle = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const isBookmarked = useEntityIsBookmarkedForQuickSwitch(entityIdentifier);
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    if (isBookmarked) {
      dispatch(bookmarkedEntityChanged({ entityIdentifier: null }));
    } else {
      dispatch(bookmarkedEntityChanged({ entityIdentifier }));
    }
  }, [dispatch, entityIdentifier, isBookmarked]);

  return (
    <IconButton
      size="sm"
      aria-label={t(isBookmarked ? 'controlLayers.removeBookmark' : 'controlLayers.bookmark')}
      tooltip={t(isBookmarked ? 'controlLayers.removeBookmark' : 'controlLayers.bookmark')}
      variant="link"
      alignSelf="stretch"
      icon={isBookmarked ? <PiBookmarkSimpleFill /> : <PiBookmarkSimpleBold />}
      onClick={onClick}
    />
  );
});

CanvasEntityIsBookmarkedForQuickSwitchToggle.displayName = 'CanvasEntityIsBookmarkedForQuickSwitchToggle';

import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { imageToCompareChanged } from 'features/gallery/store/gallerySlice';
import { isImageResource, } from 'features/gallery/store/resourceTypes';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold } from 'react-icons/pi';
import type { ImageDTO, VideoDTO } from 'services/api/types';

type Props = {
  resource: ImageDTO | VideoDTO;
  isHovered: boolean;
};

export const GalleryResourceOpenInViewerIconButton = memo(({ resource, isHovered }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    // Clear any image comparison state and open the viewer
    dispatch(imageToCompareChanged(null));
    navigationApi.focusPanelInActiveTab(VIEWER_PANEL_ID);
  }, [dispatch]);

  const tooltip = useMemo(() => {
    if (isImageResource(resource)) {
      return t('gallery.openInViewer');
    } else {
      return t('gallery.openVideoInViewer');
    }
  }, [resource, t]);

  return (
    <Tooltip label={tooltip}>
      <IconButton
        onClick={onClick}
        icon={<PiArrowsOutBold />}
        size="sm"
        variant="ghost"
        aria-label={tooltip}
        colorScheme="base"
        fontSize={14}
        position="absolute"
        bottom={1}
        insetInlineEnd={1}
        visibility={isHovered ? 'visible' : 'hidden'}
        color="base.100"
        _hover={{
          color: 'base.50',
          bg: 'base.500',
        }}
      />
    </Tooltip>
  );
});

GalleryResourceOpenInViewerIconButton.displayName = 'GalleryResourceOpenInViewerIconButton';


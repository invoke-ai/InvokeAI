import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { getResourceId, isImageResource, isVideoResource } from 'features/gallery/store/resourceTypes';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import type { ImageDTO, VideoDTO } from 'services/api/types';

type Props = {
  resource: ImageDTO | VideoDTO;
  isHovered: boolean;
};

export const GalleryResourceDeleteIconButton = memo(({ resource, isHovered }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    // For now, we'll use the existing image delete modal for both images and videos
    // Later, we can create a resource-specific delete modal
    if (isImageResource(resource)) {
      // dispatch(imageToDeleteSelected({ imageName: resource.id, imageUsage: { isCanvasImage: false, isInitialImage: false, isControlLayerImage: false, isNodesImage: false } }));
    } else if (isVideoResource(resource)) {
      // For videos, we'll need to implement video deletion
      // For now, just log that we would delete the video
      console.log('Would delete video:', getResourceId(resource));
    }
  }, [resource]);

  const tooltip = useMemo(() => {
    if (isImageResource(resource)) {
      return t('gallery.deleteImage');
    } else {
      return t('gallery.deleteVideo');
    }
  }, [resource, t]);

  return (
    <Tooltip label={tooltip}>
      <IconButton
        onClick={onClick}
        icon={<PiTrashSimpleBold />}
        size="sm"
        variant="ghost"
        aria-label={tooltip}
        colorScheme="error"
        fontSize={14}
        position="absolute"
        top={1}
        insetInlineEnd={1}
        visibility={isHovered ? 'visible' : 'hidden'}
        color="error.300"
        _hover={{
          color: 'error.400',
          bg: 'error.500',
        }}
      />
    </Tooltip>
  );
});

GalleryResourceDeleteIconButton.displayName = 'GalleryResourceDeleteIconButton';

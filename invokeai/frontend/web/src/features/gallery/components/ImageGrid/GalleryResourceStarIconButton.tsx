import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { isImageResource, isVideoResource} from 'features/gallery/store/resourceTypes';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStarBold, PiStarFill } from 'react-icons/pi';
import { useStarResourcesMutation, useUnstarResourcesMutation } from 'services/api/endpoints/resources';
import type { ImageDTO, VideoDTO } from 'services/api/types';

type Props = {
  resource: ImageDTO | VideoDTO;
  isHovered: boolean;
};

export const GalleryResourceStarIconButton = memo(({ resource, isHovered }: Props) => {
  const { t } = useTranslation();
  const [starResources] = useStarResourcesMutation();
  const [unstarResources] = useUnstarResourcesMutation();

  const toggleStarred = useCallback(() => {
    if (isImageResource(resource)) {
      if (resource.starred) {
        unstarResources({ resources: [{ resource_id: resource.image_name, resource_type: 'image' }] });
      } else {
        starResources({ resources: [{ resource_id: resource.image_name, resource_type: 'image' }] });
      }
    } else if (isVideoResource(resource)) {
      // For videos, we'll need to implement video starring
      // For now, just log that we would star/unstar the video
    }
  }, [resource, starResources, unstarResources]);

  const starred = useMemo(() => {
    if (isImageResource(resource)) {
      return resource.starred;
    } else {
      return false;
    }
  }, [resource]);

  const tooltip = useMemo(() => {
    if (starred) {
      return isImageResource(resource) ? t('gallery.unstarImage') : t('gallery.unstarVideo');
    } else {
      return isImageResource(resource) ? t('gallery.starImage') : t('gallery.starVideo');
    }
  }, [resource, t, starred]);

  return (
    <Tooltip label={tooltip}>
      <IconButton
        onClick={toggleStarred}
        icon={starred ? <PiStarFill /> : <PiStarBold />}
        size="sm"
        variant="ghost"
        aria-label={tooltip}
        colorScheme={starred ? 'invokeYellow' : 'base'}
        fontSize={14}
        position="absolute"
        top={1}
        insetInlineStart={1}
        visibility={isHovered || starred ? 'visible' : 'hidden'}
        color={starred ? 'invokeYellow.300' : 'base.100'}
        _hover={{
          color: starred ? 'invokeYellow.400' : 'base.50',
          bg: starred ? 'invokeYellow.500' : 'base.500',
        }}
      />
    </Tooltip>
  );
});

GalleryResourceStarIconButton.displayName = 'GalleryResourceStarIconButton';


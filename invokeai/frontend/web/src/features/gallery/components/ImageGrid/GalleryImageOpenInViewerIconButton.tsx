import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

type Props = {
  imageDTO: ImageDTO;
};

export const GalleryImageOpenInViewerIconButton = memo(({ imageDTO }: Props) => {
  const { t } = useTranslation();

  const onClick = useCallback(() => {
    // TODO
    imageDTO.image_name;
  }, [imageDTO]);

  return (
    <DndImageIcon
      onClick={onClick}
      icon={<PiArrowsOutBold />}
      tooltip={t('gallery.openInViewer')}
      position="absolute"
      insetBlockStart={2}
      insetInlineStart={2}
    />
  );
});

GalleryImageOpenInViewerIconButton.displayName = 'GalleryImageOpenInViewerIconButton';

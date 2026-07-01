import { MenuItem } from '@invoke-ai/ui-library';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStarBold, PiStarFill } from 'react-icons/pi';
import { useStarImagesMutation, useUnstarImagesMutation } from 'services/api/endpoints/images';

export const ContextMenuItemStarUnstar = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();

  const starImage = useCallback(() => {
    starImages({ image_names: [imageDTO.image_name] });
  }, [starImages, imageDTO]);

  const unstarImage = useCallback(() => {
    unstarImages({ image_names: [imageDTO.image_name] });
  }, [unstarImages, imageDTO]);

  if (imageDTO.starred) {
    return (
      <MenuItem icon={<PiStarFill />} onClickCapture={unstarImage}>
        {t('gallery.unstarImage')}
      </MenuItem>
    );
  }

  return (
    <MenuItem icon={<PiStarBold />} onClickCapture={starImage}>
      {t('gallery.starImage')}
    </MenuItem>
  );
});

ContextMenuItemStarUnstar.displayName = 'ContextMenuItemStarUnstar';

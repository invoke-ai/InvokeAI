import { MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStarBold, PiStarFill } from 'react-icons/pi';
import { useStarImagesMutation, useUnstarImagesMutation } from 'services/api/endpoints/images';

export const ImageMenuItemStarUnstar = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const customStarUi = useStore($customStarUI);
  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();

  const starImage = useCallback(() => {
    if (imageDTO) {
      starImages({ imageDTOs: [imageDTO] });
    }
  }, [starImages, imageDTO]);

  const unstarImage = useCallback(() => {
    if (imageDTO) {
      unstarImages({ imageDTOs: [imageDTO] });
    }
  }, [unstarImages, imageDTO]);

  if (imageDTO.starred) {
    return (
      <MenuItem icon={customStarUi ? customStarUi.off.icon : <PiStarFill />} onClickCapture={unstarImage}>
        {customStarUi ? customStarUi.off.text : t('gallery.unstarImage')}
      </MenuItem>
    );
  }

  return (
    <MenuItem icon={customStarUi ? customStarUi.on.icon : <PiStarBold />} onClickCapture={starImage}>
      {customStarUi ? customStarUi.on.text : t('gallery.starImage')}
    </MenuItem>
  );
});

ImageMenuItemStarUnstar.displayName = 'ImageMenuItemStarUnstar';

import { MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStarBold, PiStarFill } from 'react-icons/pi';
import { useStarResourcesMutation, useUnstarResourcesMutation } from 'services/api/endpoints/resources';

export const ImageMenuItemStarUnstar = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const customStarUi = useStore($customStarUI);
  const [starResources] = useStarResourcesMutation();
  const [unstarResources] = useUnstarResourcesMutation();

  const starImage = useCallback(() => {
    if (imageDTO) {
      starResources({ resources: [{ resource_id: imageDTO.image_name, resource_type: "image" }] });
    }
  }, [starResources, imageDTO]);

  const unstarImage = useCallback(() => {
    if (imageDTO) {
      unstarResources({ resources: [{ resource_id: imageDTO.image_name, resource_type: "image" }] });
    }
  }, [unstarResources, imageDTO]);

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

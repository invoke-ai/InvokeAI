import { MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useItemDTOContext } from 'features/gallery/contexts/ItemDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiStarBold, PiStarFill } from 'react-icons/pi';
import { useStarImagesMutation, useUnstarImagesMutation } from 'services/api/endpoints/images';
import { useStarVideosMutation, useUnstarVideosMutation } from 'services/api/endpoints/videos';
import { isImageDTO, isVideoDTO } from 'services/api/types';

export const ContextMenuItemStarUnstar = memo(() => {
  const { t } = useTranslation();
  const itemDTO = useItemDTOContext();
  const customStarUi = useStore($customStarUI);
  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();
  const [starVideos] = useStarVideosMutation();
  const [unstarVideos] = useUnstarVideosMutation();

  const starImage = useCallback(() => {
    if (isImageDTO(itemDTO)) {
      starImages({ image_names: [itemDTO.image_name] });
    } else if (isVideoDTO(itemDTO)) {
      starVideos({ video_ids: [itemDTO.video_id] });
    }
  }, [starImages, itemDTO, starVideos]);

  const unstarImage = useCallback(() => {
    if (isImageDTO(itemDTO)) {
      unstarImages({ image_names: [itemDTO.image_name] });
    } else if (isVideoDTO(itemDTO)) {
      unstarVideos({ video_ids: [itemDTO.video_id] });
    }
  }, [unstarImages, itemDTO, unstarVideos]);

  if (itemDTO.starred) {
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

ContextMenuItemStarUnstar.displayName = 'ContextMenuItemStarUnstar';

import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { useRecallMetadataWithConfirmation } from 'features/gallery/components/ImageGrid/RecallMetadataConfirmationAlertDialog';
import { useImageActions } from 'features/gallery/hooks/useImageActions';
import { memo, useCallback } from 'react';
import { PiAsteriskBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

type Props = {
  imageDTO: ImageDTO;
};

export const GalleryImageRecallAllIconButton = memo(({ imageDTO }: Props) => {
  const imageActions = useImageActions(imageDTO);
  const { recallWithConfirmation } = useRecallMetadataWithConfirmation();
  
  const onClick = useCallback(() => {
    if (imageActions.hasMetadata) {
      recallWithConfirmation(() => {
        imageActions.recallAll();
      });
    }
  }, [imageActions, recallWithConfirmation]);

  return (
    <DndImageIcon
      onClick={onClick}
      icon={<PiAsteriskBold />}
      tooltip="Recall"
      position="absolute"
      insetBlockStart={2}
      insetInlineStart="50%"
      transform="translateX(-50%)"
      isDisabled={!imageActions.hasMetadata}
    />
  );
});

GalleryImageRecallAllIconButton.displayName = 'GalleryImageRecallAllIconButton';

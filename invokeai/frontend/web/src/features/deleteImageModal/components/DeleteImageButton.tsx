import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { selectSelectionCount } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { $isConnected } from 'services/events/stores';

type DeleteImageButtonProps = Omit<IconButtonProps, 'aria-label'> & {
  onClick: () => void;
};

export const DeleteImageButton = memo((props: DeleteImageButtonProps) => {
  const { onClick, isDisabled } = props;
  const { t } = useTranslation();
  const isConnected = useStore($isConnected);
  const imageSelectionLength = useAppSelector(selectSelectionCount);
  const labelMessage: string = `${t('gallery.deleteImage', { count: imageSelectionLength })} (Del)`;

  return (
    <IconButton
      onClick={onClick}
      icon={<PiTrashSimpleBold />}
      tooltip={labelMessage}
      aria-label={labelMessage}
      isDisabled={isDisabled || !isConnected}
      colorScheme="error"
    />
  );
});

DeleteImageButton.displayName = 'DeleteImageButton';

import { IconButtonProps } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';

const deleteImageButtonsSelector = createSelector(
  [stateSelector],
  ({ system }) => {
    const { isProcessing, isConnected } = system;

    return isConnected && !isProcessing;
  }
);

type DeleteImageButtonProps = Omit<IconButtonProps, 'aria-label'> & {
  onClick: () => void;
};

export const DeleteImageButton = (props: DeleteImageButtonProps) => {
  const { onClick, isDisabled } = props;
  const { t } = useTranslation();
  const canDeleteImage = useAppSelector(deleteImageButtonsSelector);

  return (
    <IAIIconButton
      onClick={onClick}
      icon={<FaTrash />}
      tooltip={`${t('gallery.deleteImage')} (Del)`}
      aria-label={`${t('gallery.deleteImage')} (Del)`}
      isDisabled={isDisabled || !canDeleteImage}
      colorScheme="error"
    />
  );
};

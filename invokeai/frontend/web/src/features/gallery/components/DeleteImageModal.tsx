import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Flex,
  Text,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAISwitch from 'common/components/IAISwitch';
import { configSelector } from 'features/system/store/configSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import { setShouldConfirmOnDelete } from 'features/system/store/systemSlice';
import { isEqual } from 'lodash-es';

import { ChangeEvent, memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [systemSelector, configSelector],
  (system, config) => {
    const { shouldConfirmOnDelete } = system;
    const { canRestoreDeletedImagesFromBin } = config;
    return { shouldConfirmOnDelete, canRestoreDeletedImagesFromBin };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

interface DeleteImageModalProps {
  isOpen: boolean;
  onClose: () => void;
  handleDelete: () => void;
}

const DeleteImageModal = ({
  isOpen,
  onClose,
  handleDelete,
}: DeleteImageModalProps) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { shouldConfirmOnDelete, canRestoreDeletedImagesFromBin } =
    useAppSelector(selector);
  const cancelRef = useRef<HTMLButtonElement>(null);

  const handleChangeShouldConfirmOnDelete = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldConfirmOnDelete(!e.target.checked)),
    [dispatch]
  );

  const handleClickDelete = useCallback(() => {
    handleDelete();
    onClose();
  }, [handleDelete, onClose]);

  return (
    <AlertDialog
      isOpen={isOpen}
      leastDestructiveRef={cancelRef}
      onClose={onClose}
      isCentered
    >
      <AlertDialogOverlay>
        <AlertDialogContent>
          <AlertDialogHeader fontSize="lg" fontWeight="bold">
            {t('gallery.deleteImage')}
          </AlertDialogHeader>

          <AlertDialogBody>
            <Flex direction="column" gap={5}>
              <Flex direction="column" gap={2}>
                <Text>{t('common.areYouSure')}</Text>
                <Text>
                  {canRestoreDeletedImagesFromBin
                    ? t('gallery.deleteImageBin')
                    : t('gallery.deleteImagePermanent')}
                </Text>
              </Flex>
              <IAISwitch
                label={t('common.dontAskMeAgain')}
                isChecked={!shouldConfirmOnDelete}
                onChange={handleChangeShouldConfirmOnDelete}
              />
            </Flex>
          </AlertDialogBody>
          <AlertDialogFooter>
            <IAIButton ref={cancelRef} onClick={onClose}>
              Cancel
            </IAIButton>
            <IAIButton colorScheme="error" onClick={handleClickDelete} ml={3}>
              Delete
            </IAIButton>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialogOverlay>
    </AlertDialog>
  );
};

export default memo(DeleteImageModal);

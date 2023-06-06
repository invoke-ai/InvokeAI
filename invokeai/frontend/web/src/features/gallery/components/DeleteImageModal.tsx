import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Divider,
  Flex,
  ListItem,
  Text,
  UnorderedList,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  DeleteImageContext,
  ImageUsage,
} from 'app/contexts/DeleteImageContext';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAISwitch from 'common/components/IAISwitch';
import { configSelector } from 'features/system/store/configSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import { setShouldConfirmOnDelete } from 'features/system/store/systemSlice';
import { some } from 'lodash-es';

import { ChangeEvent, memo, useCallback, useContext, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';

const selector = createSelector(
  [systemSelector, configSelector],
  (system, config) => {
    const { shouldConfirmOnDelete } = system;
    const { canRestoreDeletedImagesFromBin } = config;

    return {
      shouldConfirmOnDelete,
      canRestoreDeletedImagesFromBin,
    };
  },
  defaultSelectorOptions
);

const ImageInUseMessage = (props: { imageUsage?: ImageUsage }) => {
  const { imageUsage } = props;

  if (!imageUsage) {
    return null;
  }

  if (!some(imageUsage)) {
    return null;
  }

  return (
    <>
      <Text>This image is currently in use in the following features:</Text>
      <UnorderedList sx={{ paddingInlineStart: 6 }}>
        {imageUsage.isInitialImage && <ListItem>Image to Image</ListItem>}
        {imageUsage.isCanvasImage && <ListItem>Unified Canvas</ListItem>}
        {imageUsage.isControlNetImage && <ListItem>ControlNet</ListItem>}
        {imageUsage.isNodesImage && <ListItem>Node Editor</ListItem>}
      </UnorderedList>
      <Text>
        If you delete this image, those features will immediately be reset.
      </Text>
    </>
  );
};

const DeleteImageModal = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { isOpen, onClose, onImmediatelyDelete, image, imageUsage } =
    useContext(DeleteImageContext);

  const { shouldConfirmOnDelete, canRestoreDeletedImagesFromBin } =
    useAppSelector(selector);

  const handleChangeShouldConfirmOnDelete = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setShouldConfirmOnDelete(!e.target.checked)),
    [dispatch]
  );

  const cancelRef = useRef<HTMLButtonElement>(null);

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
            <Flex direction="column" gap={3}>
              <ImageInUseMessage imageUsage={imageUsage} />
              <Divider />
              <Text>
                {canRestoreDeletedImagesFromBin
                  ? t('gallery.deleteImageBin')
                  : t('gallery.deleteImagePermanent')}
              </Text>
              <Text>{t('common.areYouSure')}</Text>
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
            <IAIButton colorScheme="error" onClick={onImmediatelyDelete} ml={3}>
              Delete
            </IAIButton>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialogOverlay>
    </AlertDialog>
  );
};

export default memo(DeleteImageModal);

const deleteImageButtonsSelector = createSelector(
  [systemSelector],
  (system) => {
    const { isProcessing, isConnected } = system;

    return isConnected && !isProcessing;
  }
);

type DeleteImageButtonProps = {
  onClick: () => void;
};

export const DeleteImageButton = (props: DeleteImageButtonProps) => {
  const { onClick } = props;
  const { t } = useTranslation();
  const canDeleteImage = useAppSelector(deleteImageButtonsSelector);

  return (
    <IAIIconButton
      onClick={onClick}
      icon={<FaTrash />}
      tooltip={`${t('gallery.deleteImage')} (Del)`}
      aria-label={`${t('gallery.deleteImage')} (Del)`}
      isDisabled={!canDeleteImage}
      colorScheme="error"
    />
  );
};

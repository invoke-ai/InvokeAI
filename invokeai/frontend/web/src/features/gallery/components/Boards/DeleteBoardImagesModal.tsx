import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Divider,
  Flex,
  Text,
} from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import { memo, useContext, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { DeleteBoardImagesContext } from '../../../../app/contexts/DeleteBoardImagesContext';

const DeleteBoardImagesModal = () => {
  const { t } = useTranslation();

  const {
    isOpen,
    onClose,
    board,
    handleDeleteBoardImages,
    handleDeleteBoardOnly,
  } = useContext(DeleteBoardImagesContext);

  const cancelRef = useRef<HTMLButtonElement>(null);

  return (
    <AlertDialog
      isOpen={isOpen}
      leastDestructiveRef={cancelRef}
      onClose={onClose}
      isCentered
    >
      <AlertDialogOverlay>
        {board && (
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Delete Board
            </AlertDialogHeader>

            <AlertDialogBody>
              <Flex direction="column" gap={3}>
                <Divider />
                <Text>{t('common.areYouSure')}</Text>
                <Text fontWeight="bold">
                  This board has {board.image_count} image(s) that will be
                  deleted.
                </Text>
              </Flex>
            </AlertDialogBody>
            <AlertDialogFooter gap={3}>
              <IAIButton ref={cancelRef} onClick={onClose}>
                Cancel
              </IAIButton>
              <IAIButton
                colorScheme="warning"
                onClick={() => handleDeleteBoardOnly(board.board_id)}
              >
                Delete Board Only
              </IAIButton>
              <IAIButton
                colorScheme="error"
                onClick={() => handleDeleteBoardImages(board.board_id)}
              >
                Delete Board and Images
              </IAIButton>
            </AlertDialogFooter>
          </AlertDialogContent>
        )}
      </AlertDialogOverlay>
    </AlertDialog>
  );
};

export default memo(DeleteBoardImagesModal);

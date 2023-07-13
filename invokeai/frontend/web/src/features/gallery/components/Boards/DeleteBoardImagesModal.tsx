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
import IAIButton from 'common/components/IAIButton';
import { memo, useContext, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { DeleteBoardImagesContext } from '../../../../app/contexts/DeleteBoardImagesContext';
import { some } from 'lodash-es';
import { ImageUsage } from '../../../../app/contexts/DeleteImageContext';

const BoardImageInUseMessage = (props: { imagesUsage?: ImageUsage }) => {
  const { imagesUsage } = props;

  if (!imagesUsage) {
    return null;
  }

  if (!some(imagesUsage)) {
    return null;
  }

  return (
    <>
      <Text>
        An image from this board is currently in use in the following features:
      </Text>
      <UnorderedList sx={{ paddingInlineStart: 6 }}>
        {imagesUsage.isInitialImage && <ListItem>Image to Image</ListItem>}
        {imagesUsage.isCanvasImage && <ListItem>Unified Canvas</ListItem>}
        {imagesUsage.isControlNetImage && <ListItem>ControlNet</ListItem>}
        {imagesUsage.isNodesImage && <ListItem>Node Editor</ListItem>}
      </UnorderedList>
      <Text>
        If you delete images from this board, those features will immediately be
        reset.
      </Text>
    </>
  );
};

const DeleteBoardImagesModal = () => {
  const { t } = useTranslation();

  const {
    isOpen,
    onClose,
    board,
    handleDeleteBoardImages,
    handleDeleteBoardOnly,
    imagesUsage,
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
                <BoardImageInUseMessage imagesUsage={imagesUsage} />
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

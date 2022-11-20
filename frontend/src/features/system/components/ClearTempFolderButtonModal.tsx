import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Button,
  Flex,
  useDisclosure,
} from '@chakra-ui/react';
import { emptyTempFolder } from 'app/socketio/actions';
import { useAppDispatch } from 'app/store';
import IAIButton from 'common/components/IAIButton';
import {
  clearCanvasHistory,
  resetCanvas,
} from 'features/canvas/store/canvasSlice';
import { useRef } from 'react';
import { FaTrash } from 'react-icons/fa';

const ClearTempFolderButtonModal = () => {
  const dispatch = useAppDispatch();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const cancelRef = useRef<HTMLButtonElement | null>(null);

  const handleClear = () => {
    dispatch(emptyTempFolder());
    dispatch(clearCanvasHistory());
    dispatch(resetCanvas());
    onClose();
  };

  return (
    <>
      <IAIButton leftIcon={<FaTrash />} size={'sm'} onClick={onOpen}>
        Clear Temp Image Folder
      </IAIButton>

      <AlertDialog
        isOpen={isOpen}
        leastDestructiveRef={cancelRef}
        onClose={onClose}
      >
        <AlertDialogOverlay>
          <AlertDialogContent className="modal">
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Clear Temp Image Folder
            </AlertDialogHeader>

            <AlertDialogBody>
              <p>
                Clearing the temp image folder also fully resets the Unified
                Canvas. This includes all undo/redo history, images in the
                staging area, and the canvas base layer.
              </p>
              <br />
              <p>Are you sure you want to clear the temp folder?</p>
            </AlertDialogBody>

            <AlertDialogFooter>
              <Button
                ref={cancelRef}
                onClick={onClose}
                className="modal-close-btn"
              >
                Cancel
              </Button>
              <Button colorScheme="red" onClick={handleClear} ml={3}>
                Clear
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </>
  );
};
export default ClearTempFolderButtonModal;

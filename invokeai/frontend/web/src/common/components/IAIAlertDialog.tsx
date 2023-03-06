import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Button,
  forwardRef,
  useDisclosure,
} from '@chakra-ui/react';
import { cloneElement, ReactElement, ReactNode, useRef } from 'react';

type Props = {
  acceptButtonText?: string;
  acceptCallback: () => void;
  cancelButtonText?: string;
  cancelCallback?: () => void;
  children: ReactNode;
  title: string;
  triggerComponent: ReactElement;
};

const IAIAlertDialog = forwardRef((props: Props, ref) => {
  const {
    acceptButtonText = 'Accept',
    acceptCallback,
    cancelButtonText = 'Cancel',
    cancelCallback,
    children,
    title,
    triggerComponent,
  } = props;

  const { isOpen, onOpen, onClose } = useDisclosure();
  const cancelRef = useRef<HTMLButtonElement | null>(null);

  const handleAccept = () => {
    acceptCallback();
    onClose();
  };

  const handleCancel = () => {
    cancelCallback && cancelCallback();
    onClose();
  };

  return (
    <>
      {cloneElement(triggerComponent, {
        onClick: onOpen,
        ref: ref,
      })}

      <AlertDialog
        isOpen={isOpen}
        leastDestructiveRef={cancelRef}
        onClose={onClose}
      >
        <AlertDialogOverlay>
          <AlertDialogContent className="modal">
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              {title}
            </AlertDialogHeader>

            <AlertDialogBody>{children}</AlertDialogBody>

            <AlertDialogFooter>
              <Button
                ref={cancelRef}
                onClick={handleCancel}
                className="modal-close-btn"
              >
                {cancelButtonText}
              </Button>
              <Button colorScheme="red" onClick={handleAccept} ml={3}>
                {acceptButtonText}
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </>
  );
});
export default IAIAlertDialog;

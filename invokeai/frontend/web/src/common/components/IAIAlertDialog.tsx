import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  forwardRef,
  useDisclosure,
} from '@chakra-ui/react';
import { cloneElement, ReactElement, ReactNode, useRef } from 'react';
import IAIButton from './IAIButton';

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
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              {title}
            </AlertDialogHeader>

            <AlertDialogBody>{children}</AlertDialogBody>

            <AlertDialogFooter>
              <IAIButton ref={cancelRef} onClick={handleCancel}>
                {cancelButtonText}
              </IAIButton>
              <IAIButton colorScheme="error" onClick={handleAccept} ml={3}>
                {acceptButtonText}
              </IAIButton>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </>
  );
});
export default IAIAlertDialog;

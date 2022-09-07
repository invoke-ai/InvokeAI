import {
  IconButton,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Text,
  Tooltip,
  useDisclosure,
} from '@chakra-ui/react';
import { MdDeleteForever } from 'react-icons/md';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { useSocketIOEmitters } from '../../app/socket';
import { RootState } from '../../app/store';
import SDButton from '../../components/SDButton';
import { setShouldConfirmOnDelete } from '../system/systemSlice';

type Props = {
  uuid: string;
};

const DeleteImageModalButton = ({ uuid }: Props) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { emitDeleteImage } = useSocketIOEmitters();
  const dispatch = useAppDispatch();
  const { shouldConfirmOnDelete } = useAppSelector(
    (state: RootState) => state.system
  );

  const handleDelete = () => {
    emitDeleteImage(uuid);
    onClose();
  };

  const handleDeleteAndDontAsk = () => {
    emitDeleteImage(uuid);
    dispatch(setShouldConfirmOnDelete(false));
    onClose();
  };

  return (
    <>
      <Tooltip label='Delete'>
        <IconButton
          aria-label='Delete'
          icon={<MdDeleteForever />}
          fontSize={24}
          onClick={shouldConfirmOnDelete ? onOpen : handleDelete}
        />
      </Tooltip>

      <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Are you sure you want to delete this image?</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Text>It will be deleted forever!</Text>
          </ModalBody>

          <ModalFooter justifyContent={'space-between'}>
            <SDButton label={'Yes'} colorScheme='red' onClick={handleDelete} />
            <SDButton
              label={"Yes, and don't ask me again"}
              colorScheme='red'
              onClick={handleDeleteAndDontAsk}
            />
            <SDButton label='Cancel' colorScheme='blue' onClick={onClose} />
          </ModalFooter>
        </ModalContent>
      </Modal>
    </>
  );
};

export default DeleteImageModalButton;

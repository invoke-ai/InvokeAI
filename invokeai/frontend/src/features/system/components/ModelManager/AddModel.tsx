import {
  Flex,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  Text,
  useDisclosure,
} from '@chakra-ui/react';

import IAIButton from 'common/components/IAIButton';

import { FaPlus } from 'react-icons/fa';

import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { useTranslation } from 'react-i18next';

import type { RootState } from 'app/store';
import { setAddNewModelUIOption } from 'features/ui/store/uiSlice';
import AddCheckpointModel from './AddCheckpointModel';
import AddDiffusersModel from './AddDiffusersModel';

function AddModelBox({
  text,
  onClick,
}: {
  text: string;
  onClick?: () => void;
}) {
  return (
    <Flex
      position="relative"
      width="50%"
      height="200px"
      backgroundColor="var(--background-color)"
      borderRadius="0.5rem"
      justifyContent="center"
      alignItems="center"
      _hover={{
        cursor: 'pointer',
        backgroundColor: 'var(--accent-color)',
      }}
      onClick={onClick}
    >
      <Text fontWeight="bold">{text}</Text>
    </Flex>
  );
}

export default function AddModel() {
  const { isOpen, onOpen, onClose } = useDisclosure();

  const addNewModelUIOption = useAppSelector(
    (state: RootState) => state.ui.addNewModelUIOption
  );

  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  const addModelModalClose = () => {
    onClose();
    dispatch(setAddNewModelUIOption(null));
  };

  return (
    <>
      <IAIButton
        aria-label={t('modelManager.addNewModel')}
        tooltip={t('modelManager.addNewModel')}
        onClick={onOpen}
        className="modal-close-btn"
        size="sm"
      >
        <Flex columnGap="0.5rem" alignItems="center">
          <FaPlus />
          {t('modelManager.addNew')}
        </Flex>
      </IAIButton>

      <Modal
        isOpen={isOpen}
        onClose={addModelModalClose}
        size="3xl"
        closeOnOverlayClick={false}
      >
        <ModalOverlay />
        <ModalContent
          className="modal add-model-modal"
          fontFamily="Inter"
          margin="auto"
        >
          <ModalHeader>{t('modelManager.addNewModel')}</ModalHeader>
          <ModalCloseButton marginTop="0.3rem" />
          <ModalBody className="add-model-modal-body">
            {addNewModelUIOption == null && (
              <Flex columnGap="1rem">
                <AddModelBox
                  text={t('modelManager.addCheckpointModel')}
                  onClick={() => dispatch(setAddNewModelUIOption('ckpt'))}
                />
                <AddModelBox
                  text={t('modelManager.addDiffuserModel')}
                  onClick={() => dispatch(setAddNewModelUIOption('diffusers'))}
                />
              </Flex>
            )}
            {addNewModelUIOption == 'ckpt' && <AddCheckpointModel />}
            {addNewModelUIOption == 'diffusers' && <AddDiffusersModel />}
          </ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
}

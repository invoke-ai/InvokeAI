import {
  Button,
  Flex,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Text,
  useDisclosure,
} from '@chakra-ui/react';

import IAIButton from 'common/components/IAIButton';

import { FaArrowLeft, FaPlus } from 'react-icons/fa';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useTranslation } from 'react-i18next';

import type { RootState } from 'app/store/store';
import { setAddNewModelUIOption } from 'features/ui/store/uiSlice';
import AddCheckpointModel from './AddCheckpointModel';
import AddDiffusersModel from './AddDiffusersModel';
import IAIIconButton from 'common/components/IAIIconButton';

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
      height={40}
      justifyContent="center"
      alignItems="center"
      onClick={onClick}
      as={Button}
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
        size="sm"
      >
        <Flex columnGap={2} alignItems="center">
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
        <ModalContent margin="auto">
          <ModalHeader>{t('modelManager.addNewModel')} </ModalHeader>
          {addNewModelUIOption !== null && (
            <IAIIconButton
              aria-label={t('common.back')}
              tooltip={t('common.back')}
              onClick={() => dispatch(setAddNewModelUIOption(null))}
              position="absolute"
              variant="ghost"
              zIndex={1}
              size="sm"
              insetInlineEnd={12}
              top={2}
              icon={<FaArrowLeft />}
            />
          )}
          <ModalCloseButton />
          <ModalBody>
            {addNewModelUIOption == null && (
              <Flex columnGap={4}>
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
          <ModalFooter />
        </ModalContent>
      </Modal>
    </>
  );
}

import {
  Flex,
  Modal,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  useDisclosure,
} from '@chakra-ui/react';
import React, { cloneElement } from 'react';

import { useTranslation } from 'react-i18next';
import { useAppSelector } from 'app/storeHooks';
import { RootState } from 'app/store';

import type { ReactElement } from 'react';

import ModelList from './ModelList';
import DiffusersModelEdit from './DiffusersModelEdit';
import CheckpointModelEdit from './CheckpointModelEdit';

type ModelManagerModalProps = {
  children: ReactElement;
};

export default function ModelManagerModal({
  children,
}: ModelManagerModalProps) {
  const {
    isOpen: isModelManagerModalOpen,
    onOpen: onModelManagerModalOpen,
    onClose: onModelManagerModalClose,
  } = useDisclosure();

  const model_list = useAppSelector(
    (state: RootState) => state.system.model_list
  );

  const openModel = useAppSelector(
    (state: RootState) => state.system.openModel
  );

  const { t } = useTranslation();

  return (
    <>
      {cloneElement(children, {
        onClick: onModelManagerModalOpen,
      })}
      <Modal
        isOpen={isModelManagerModalOpen}
        onClose={onModelManagerModalClose}
        size="6xl"
      >
        <ModalOverlay />
        <ModalContent className="modal" fontFamily="Inter">
          <ModalCloseButton className="modal-close-btn" />
          <ModalHeader fontWeight="bold">
            {t('modelmanager:modelManager')}
          </ModalHeader>
          <Flex
            padding={'0 1.5rem 1.5rem 1.5rem'}
            width="100%"
            columnGap={'2rem'}
          >
            <ModelList />
            {openModel && model_list[openModel]['format'] === 'diffusers' ? (
              <DiffusersModelEdit />
            ) : (
              <CheckpointModelEdit />
            )}
          </Flex>
        </ModalContent>
      </Modal>
    </>
  );
}

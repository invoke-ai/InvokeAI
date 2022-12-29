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

import ModelEdit from './ModelEdit';
import ModelList from './ModelList';

import type { ReactElement } from 'react';

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
        <ModalContent className=" modal">
          <ModalCloseButton className="modal-close-btn" />
          <ModalHeader>{t('modelmanager:modelManager')}</ModalHeader>
          <Flex
            padding={'0 1.5rem 1.5rem 1.5rem'}
            width="100%"
            columnGap={'2rem'}
          >
            <ModelList />
            <ModelEdit />
          </Flex>
        </ModalContent>
      </Modal>
    </>
  );
}

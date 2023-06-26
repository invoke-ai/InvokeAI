import {
  Flex,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  useDisclosure,
} from '@chakra-ui/react';
import { cloneElement } from 'react';

import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useTranslation } from 'react-i18next';

import type { ReactElement } from 'react';

import { useListModelsQuery } from 'services/api/endpoints/models';
import CheckpointModelEdit from './CheckpointModelEdit';
import DiffusersModelEdit from './DiffusersModelEdit';
import ModelList from './ModelList';

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

  const { data: pipelineModels } = useListModelsQuery({
    model_type: 'pipeline',
  });

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
        size="full"
      >
        <ModalOverlay />
        <ModalContent>
          <ModalCloseButton />
          <ModalHeader>{t('modelManager.modelManager')}</ModalHeader>
          <ModalBody>
            <Flex width="100%" columnGap={8}>
              <ModelList />
              {openModel &&
              pipelineModels &&
              pipelineModels['entities'][openModel]['model_format'] ===
                'diffusers' ? (
                <DiffusersModelEdit />
              ) : (
                <CheckpointModelEdit />
              )}
            </Flex>
          </ModalBody>
          <ModalFooter />
        </ModalContent>
      </Modal>
    </>
  );
}

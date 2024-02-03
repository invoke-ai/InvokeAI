import {
  Flex,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
} from '@invoke-ai/ui-library';
import { useDynamicPromptsModal } from 'features/dynamicPrompts/hooks/useDynamicPromptsModal';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import ParamDynamicPromptsMaxPrompts from './ParamDynamicPromptsMaxPrompts';
import ParamDynamicPromptsPreview from './ParamDynamicPromptsPreview';
import ParamDynamicPromptsSeedBehaviour from './ParamDynamicPromptsSeedBehaviour';

export const DynamicPromptsModal = memo(() => {
  const { t } = useTranslation();
  const { isOpen, onClose } = useDynamicPromptsModal();

  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered>
      <ModalOverlay />
      <ModalContent w="80vw" h="80vh" maxW="unset" maxH="unset">
        <ModalHeader>{t('dynamicPrompts.dynamicPrompts')}</ModalHeader>
        <ModalCloseButton />
        <ModalBody as={Flex} flexDir="column" gap={4} w="full" h="full" pb={4}>
          <Flex gap={4}>
            <ParamDynamicPromptsSeedBehaviour />
            <ParamDynamicPromptsMaxPrompts />
          </Flex>
          <ParamDynamicPromptsPreview />
        </ModalBody>
      </ModalContent>
    </Modal>
  );
});

DynamicPromptsModal.displayName = 'DynamicPromptsModal';

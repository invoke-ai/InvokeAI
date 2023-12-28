import { Flex } from '@chakra-ui/layout';
import {
  InvModal,
  InvModalBody,
  InvModalCloseButton,
  InvModalContent,
  InvModalHeader,
  InvModalOverlay,
} from 'common/components/InvModal/wrapper';
import { useDynamicPromptsModal } from 'features/dynamicPrompts/hooks/useDynamicPromptsModal';
import { useTranslation } from 'react-i18next';

import ParamDynamicPromptsMaxPrompts from './ParamDynamicPromptsMaxPrompts';
import ParamDynamicPromptsPreview from './ParamDynamicPromptsPreview';
import ParamDynamicPromptsSeedBehaviour from './ParamDynamicPromptsSeedBehaviour';

export const DynamicPromptsModal = () => {
  const { t } = useTranslation();
  const { isOpen, onClose } = useDynamicPromptsModal();

  return (
    <InvModal isOpen={isOpen} onClose={onClose} isCentered>
      <InvModalOverlay />
      <InvModalContent w="80vw" h="80vh" maxW="unset" maxH="unset">
        <InvModalHeader>{t('dynamicPrompts.dynamicPrompts')}</InvModalHeader>
        <InvModalCloseButton />
        <InvModalBody
          as={Flex}
          flexDir="column"
          gap={2}
          w="full"
          h="full"
          pb={4}
        >
          <Flex gap={4}>
            <ParamDynamicPromptsSeedBehaviour />
            <ParamDynamicPromptsMaxPrompts />
          </Flex>
          <ParamDynamicPromptsPreview />
        </InvModalBody>
      </InvModalContent>
    </InvModal>
  );
};

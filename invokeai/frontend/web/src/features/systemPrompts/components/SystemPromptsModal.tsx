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
  Spinner,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { SystemPromptForm } from 'features/systemPrompts/components/SystemPromptForm';
import { SystemPromptListItem } from 'features/systemPrompts/components/SystemPromptListItem';
import {
  $systemPromptsModalState,
  closeSystemPromptsModal,
  showSystemPromptEditor,
} from 'features/systemPrompts/store/systemPromptModal';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { useListSystemPromptsQuery } from 'services/api/endpoints/systemPrompts';

export const SystemPromptsModal = memo(() => {
  useAssertSingleton('SystemPromptsModal');
  const { t } = useTranslation();
  const state = useStore($systemPromptsModalState);
  const { data: prompts, isLoading } = useListSystemPromptsQuery(undefined, { skip: !state.isOpen });

  const editingPrompt = useMemo(() => {
    if (!state.editingId || state.editingId === 'new') {
      return null;
    }
    return prompts?.find((p) => p.id === state.editingId) ?? null;
  }, [state.editingId, prompts]);

  const isEditMode = state.editingId !== null;

  const title = isEditMode
    ? state.editingId === 'new'
      ? t('systemPrompts.newSystemPrompt')
      : t('systemPrompts.editSystemPrompt')
    : t('systemPrompts.manageSystemPrompts');

  const handleNew = useCallback(() => {
    showSystemPromptEditor('new');
  }, []);

  return (
    <Modal isOpen={state.isOpen} onClose={closeSystemPromptsModal} isCentered size="4xl" useInert={false}>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>{title}</ModalHeader>
        <ModalCloseButton />
        <ModalBody display="flex" flexDir="column" gap={4}>
          {isEditMode ? (
            <SystemPromptForm editing={editingPrompt} />
          ) : isLoading ? (
            <Spinner alignSelf="center" />
          ) : !prompts || prompts.length === 0 ? (
            <Text color="base.300" alignSelf="center" py={6}>
              {t('systemPrompts.noPromptsYet')}
            </Text>
          ) : (
            <Flex flexDir="column" gap={2}>
              {prompts.map((p) => (
                <SystemPromptListItem key={p.id} prompt={p} />
              ))}
            </Flex>
          )}
        </ModalBody>
        <ModalFooter p={4}>
          {!isEditMode && (
            <Button leftIcon={<PiPlusBold />} colorScheme="invokeBlue" onClick={handleNew}>
              {t('systemPrompts.newSystemPrompt')}
            </Button>
          )}
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
});

SystemPromptsModal.displayName = 'SystemPromptsModal';

import { Badge, Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { useDeleteSystemPrompt } from 'features/systemPrompts/components/DeleteSystemPromptDialog';
import { useCanEditSystemPrompt } from 'features/systemPrompts/hooks/useCanEditSystemPrompt';
import { showSystemPromptEditor } from 'features/systemPrompts/store/systemPromptModal';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPencilSimpleBold, PiTrashBold } from 'react-icons/pi';
import type { SystemPromptRecordDTO } from 'services/api/endpoints/systemPrompts';

const SYSTEM_USER_ID = 'system';

type Props = {
  prompt: SystemPromptRecordDTO;
};

export const SystemPromptListItem = memo(({ prompt }: Props) => {
  const { t } = useTranslation();
  const deletePrompt = useDeleteSystemPrompt();
  const canEdit = useCanEditSystemPrompt(prompt);

  const handleEdit = useCallback(() => {
    showSystemPromptEditor(prompt.id);
  }, [prompt.id]);

  const handleDelete = useCallback(() => {
    deletePrompt(prompt);
  }, [deletePrompt, prompt]);

  const isSystemDefault = prompt.user_id === SYSTEM_USER_ID;

  return (
    <Flex flexDir="column" gap={1} p={3} borderRadius="base" bg="base.800" _hover={{ bg: 'base.750' }}>
      <Flex alignItems="center" gap={2}>
        <Text fontWeight="semibold" noOfLines={1}>
          {prompt.name}
        </Text>
        {isSystemDefault && (
          <Badge colorScheme="invokeBlue" variant="subtle">
            {t('systemPrompts.systemBadge')}
          </Badge>
        )}
        {!isSystemDefault && prompt.is_public && (
          <Badge colorScheme="green" variant="subtle">
            {t('systemPrompts.sharedBadge')}
          </Badge>
        )}
        <Spacer />
        {canEdit && (
          <>
            <IconButton
              aria-label={t('common.edit')}
              tooltip={t('common.edit')}
              icon={<PiPencilSimpleBold />}
              size="sm"
              variant="ghost"
              onClick={handleEdit}
            />
            <IconButton
              aria-label={t('common.delete')}
              tooltip={t('common.delete')}
              icon={<PiTrashBold />}
              size="sm"
              variant="ghost"
              colorScheme="error"
              onClick={handleDelete}
            />
          </>
        )}
      </Flex>
      <Text fontSize="sm" color="base.300" noOfLines={2}>
        {prompt.content}
      </Text>
    </Flex>
  );
});

SystemPromptListItem.displayName = 'SystemPromptListItem';

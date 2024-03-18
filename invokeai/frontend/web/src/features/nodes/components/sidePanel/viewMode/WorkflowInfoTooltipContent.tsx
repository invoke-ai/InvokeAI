import { Box, Flex, Text } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => {
  return {
    name: workflow.name,
    description: workflow.description,
    notes: workflow.notes,
    author: workflow.author,
    tags: workflow.tags,
  };
});

const WorkflowInfoTooltipContent = () => {
  const { name, description, notes, author, tags } = useAppSelector(selector);
  const { t } = useTranslation();

  return (
    <Flex flexDir="column" gap="2">
      {!!name.length && (
        <Box>
          <Text fontWeight="semibold">{t('nodes.workflowName')}</Text>
          <Text opacity={0.7} fontStyle="oblique 5deg">
            {name}
          </Text>
        </Box>
      )}
      {!!author.length && (
        <Box>
          <Text fontWeight="semibold">{t('nodes.workflowAuthor')}</Text>
          <Text opacity={0.7} fontStyle="oblique 5deg">
            {author}
          </Text>
        </Box>
      )}
      {!!tags.length && (
        <Box>
          <Text fontWeight="semibold">{t('nodes.workflowTags')}</Text>
          <Text opacity={0.7} fontStyle="oblique 5deg">
            {tags}
          </Text>
        </Box>
      )}
      {!!description.length && (
        <Box>
          <Text fontWeight="semibold">{t('nodes.workflowDescription')}</Text>
          <Text opacity={0.7} fontStyle="oblique 5deg">
            {description}
          </Text>
        </Box>
      )}
      {!!notes.length && (
        <Box>
          <Text fontWeight="semibold">{t('nodes.workflowNotes')}</Text>
          <Text opacity={0.7} fontStyle="oblique 5deg">
            {notes}
          </Text>
        </Box>
      )}
    </Flex>
  );
};

export default memo(WorkflowInfoTooltipContent);

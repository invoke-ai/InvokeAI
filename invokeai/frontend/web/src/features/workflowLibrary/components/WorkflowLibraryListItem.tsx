import { Flex, Heading, Spacer, Text } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import dateFormat, { masks } from 'dateformat';
import { useDeleteLibraryWorkflow } from 'features/workflowLibrary/hooks/useDeleteLibraryWorkflow';
import { useGetAndLoadLibraryWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadLibraryWorkflow';
import { useWorkflowLibraryModalContext } from 'features/workflowLibrary/context/useWorkflowLibraryModalContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { WorkflowRecordListItemDTO } from 'services/api/types';

type Props = {
  workflowDTO: WorkflowRecordListItemDTO;
};

const WorkflowLibraryListItem = ({ workflowDTO }: Props) => {
  const { t } = useTranslation();
  const { onClose } = useWorkflowLibraryModalContext();
  const { deleteWorkflow, deleteWorkflowResult } = useDeleteLibraryWorkflow({});
  const { getAndLoadWorkflow, getAndLoadWorkflowResult } =
    useGetAndLoadLibraryWorkflow({ onSuccess: onClose });

  const handleDeleteWorkflow = useCallback(() => {
    deleteWorkflow(workflowDTO.workflow_id);
  }, [deleteWorkflow, workflowDTO.workflow_id]);

  const handleGetAndLoadWorkflow = useCallback(() => {
    getAndLoadWorkflow(workflowDTO.workflow_id);
  }, [getAndLoadWorkflow, workflowDTO.workflow_id]);

  return (
    <Flex key={workflowDTO.workflow_id} w="full">
      <Flex w="full" alignItems="center" gap={2}>
        <Flex flexDir="column" flexGrow={1}>
          <Flex alignItems="center" w="full">
            <Heading size="sm">
              {workflowDTO.name || t('workflows.unnamedWorkflow')}
            </Heading>
            <Spacer />
            {workflowDTO.category === 'user' && (
              <Text fontSize="sm" variant="subtext">
                {t('common.updated')}:{' '}
                {dateFormat(workflowDTO.updated_at, masks.shortDate)}{' '}
                {dateFormat(workflowDTO.updated_at, masks.shortTime)}
              </Text>
            )}
          </Flex>
          <Flex alignItems="center" w="full">
            {workflowDTO.description ? (
              <Text fontSize="sm" noOfLines={1}>
                {workflowDTO.description}
              </Text>
            ) : (
              <Text
                fontSize="sm"
                variant="subtext"
                fontStyle="italic"
                noOfLines={1}
              >
                {t('workflows.noDescription')}
              </Text>
            )}
            <Spacer />
            {workflowDTO.category === 'user' && (
              <Text fontSize="sm" variant="subtext">
                {t('common.created')}:{' '}
                {dateFormat(workflowDTO.created_at, masks.shortDate)}{' '}
                {dateFormat(workflowDTO.created_at, masks.shortTime)}
              </Text>
            )}
          </Flex>
        </Flex>
        <IAIButton
          onClick={handleGetAndLoadWorkflow}
          isLoading={getAndLoadWorkflowResult.isLoading}
          aria-label={t('workflows.openWorkflow')}
        >
          {t('common.load')}
        </IAIButton>
        {workflowDTO.category === 'user' && (
          <IAIButton
            colorScheme="error"
            onClick={handleDeleteWorkflow}
            isLoading={deleteWorkflowResult.isLoading}
            aria-label={t('workflows.deleteWorkflow')}
          >
            {t('common.delete')}
          </IAIButton>
        )}
      </Flex>
    </Flex>
  );
};

export default memo(WorkflowLibraryListItem);

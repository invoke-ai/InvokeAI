import { Flex, Heading, Spacer, Text } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import dateFormat from 'dateformat';
import { useWorkflowLibraryContext } from 'features/nodes/components/flow/WorkflowLibrary/useWorkflowLibraryContext';
import { useDeleteLibraryWorkflow } from 'features/nodes/hooks/useDeleteLibraryWorkflow';
import { useGetAndLoadLibraryWorkflow } from 'features/nodes/hooks/useGetAndLoadLibraryWorkflow';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { paths } from 'services/api/schema';

type Props = {
  workflowDTO: paths['/api/v1/workflows/']['get']['responses']['200']['content']['application/json']['items'][number];
};

const WorkflowLibraryList = ({ workflowDTO }: Props) => {
  const { t } = useTranslation();
  const { onClose } = useWorkflowLibraryContext();
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
            <Text fontSize="sm" fontStyle="italic" variant="subtext">
              {t('common.lastUpdated', {
                date: dateFormat(workflowDTO.updated_at),
              })}
            </Text>
          </Flex>
          <Text fontSize="sm" noOfLines={1}>
            {workflowDTO.description}
          </Text>
        </Flex>
        <IAIButton
          onClick={handleGetAndLoadWorkflow}
          isLoading={getAndLoadWorkflowResult.isLoading}
          aria-label={t('workflows.loadWorkflow')}
        >
          {t('common.load')}
        </IAIButton>
        <IAIButton
          colorScheme="error"
          onClick={handleDeleteWorkflow}
          isLoading={deleteWorkflowResult.isLoading}
          aria-label={t('workflows.deleteWorkflow')}
        >
          {t('common.delete')}
        </IAIButton>
      </Flex>
    </Flex>
  );
};

export default memo(WorkflowLibraryList);

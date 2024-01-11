import { Flex, Heading, Spacer } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvText } from 'common/components/InvText/wrapper';
import dateFormat, { masks } from 'dateformat';
import { useWorkflowLibraryModalContext } from 'features/workflowLibrary/context/useWorkflowLibraryModalContext';
import { useDeleteLibraryWorkflow } from 'features/workflowLibrary/hooks/useDeleteLibraryWorkflow';
import { useGetAndLoadLibraryWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadLibraryWorkflow';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { WorkflowRecordListItemDTO } from 'services/api/types';

type Props = {
  workflowDTO: WorkflowRecordListItemDTO;
};

const WorkflowLibraryListItem = ({ workflowDTO }: Props) => {
  const { t } = useTranslation();
  const workflowId = useAppSelector((s) => s.workflow.id);
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

  const isOpen = useMemo(
    () => workflowId === workflowDTO.workflow_id,
    [workflowId, workflowDTO.workflow_id]
  );

  return (
    <Flex
      key={workflowDTO.workflow_id}
      w="full"
      p={2}
      borderRadius="base"
      _hover={{ bg: 'base.750' }}
    >
      <Flex w="full" alignItems="center" gap={2} h={12}>
        <Flex flexDir="column" flexGrow={1} h="full">
          <Flex alignItems="center" w="full" h="50%">
            <Heading
              size="sm"
              noOfLines={1}
              variant={isOpen ? 'invokeBlue' : undefined}
            >
              {workflowDTO.name || t('workflows.unnamedWorkflow')}
            </Heading>
            <Spacer />
            {workflowDTO.category === 'user' && (
              <InvText
                fontSize="sm"
                variant="subtext"
                flexShrink={0}
                noOfLines={1}
              >
                {t('common.updated')}:{' '}
                {dateFormat(workflowDTO.updated_at, masks.shortDate)}{' '}
                {dateFormat(workflowDTO.updated_at, masks.shortTime)}
              </InvText>
            )}
          </Flex>
          <Flex alignItems="center" w="full" h="50%">
            {workflowDTO.description ? (
              <InvText fontSize="sm" noOfLines={1}>
                {workflowDTO.description}
              </InvText>
            ) : (
              <InvText
                fontSize="sm"
                variant="subtext"
                fontStyle="italic"
                noOfLines={1}
              >
                {t('workflows.noDescription')}
              </InvText>
            )}
            <Spacer />
            {workflowDTO.category === 'user' && (
              <InvText
                fontSize="sm"
                variant="subtext"
                flexShrink={0}
                noOfLines={1}
              >
                {t('common.created')}:{' '}
                {dateFormat(workflowDTO.created_at, masks.shortDate)}{' '}
                {dateFormat(workflowDTO.created_at, masks.shortTime)}
              </InvText>
            )}
          </Flex>
        </Flex>
        <InvButton
          flexShrink={0}
          isDisabled={isOpen}
          onClick={handleGetAndLoadWorkflow}
          isLoading={getAndLoadWorkflowResult.isLoading}
          aria-label={t('workflows.openWorkflow')}
        >
          {t('common.load')}
        </InvButton>
        {workflowDTO.category === 'user' && (
          <InvButton
            flexShrink={0}
            colorScheme="error"
            isDisabled={isOpen}
            onClick={handleDeleteWorkflow}
            isLoading={deleteWorkflowResult.isLoading}
            aria-label={t('workflows.deleteWorkflow')}
          >
            {t('common.delete')}
          </InvButton>
        )}
      </Flex>
    </Flex>
  );
};

export default memo(WorkflowLibraryListItem);

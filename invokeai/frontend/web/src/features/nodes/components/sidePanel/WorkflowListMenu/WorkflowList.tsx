import { Button, Collapse, Flex, Icon, Spinner, Text } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { useCategorySections } from 'features/nodes/hooks/useCategorySections';
import {
  selectWorkflowOrderBy,
  selectWorkflowOrderDirection,
  selectWorkflowSearchTerm,
} from 'features/nodes/store/workflowSlice';
import type { WorkflowCategory } from 'features/nodes/types/workflow';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';
import { useListWorkflowsQuery } from 'services/api/endpoints/workflows';

import { WorkflowListItem } from './WorkflowListItem';

export const WorkflowList = ({ category }: { category: WorkflowCategory }) => {
  const searchTerm = useAppSelector(selectWorkflowSearchTerm);
  const orderBy = useAppSelector(selectWorkflowOrderBy);
  const direction = useAppSelector(selectWorkflowOrderDirection);
  const { t } = useTranslation();

  const queryArg = useMemo<Parameters<typeof useListWorkflowsQuery>[0]>(() => {
    if (category !== 'default') {
      return {
        order_by: orderBy,
        direction,
        category: category,
      };
    }
    return {
      order_by: 'name' as const,
      direction: 'ASC' as const,
      category: category,
    };
  }, [category, direction, orderBy]);

  const { data, isLoading } = useListWorkflowsQuery(queryArg, {
    selectFromResult: ({ data, isLoading }) => {
      const filteredData =
        data?.items.filter((workflow) => workflow.name.toLowerCase().includes(searchTerm.toLowerCase())) || EMPTY_ARRAY;

      return {
        data: filteredData,
        isLoading,
      };
    },
  });

  const { isOpen, onToggle } = useCategorySections(category);

  return (
    <Flex flexDir="column">
      <Button variant="unstyled" onClick={onToggle}>
        <Flex gap={2} alignItems="center">
          <Icon boxSize={4} as={PiCaretDownBold} transform={isOpen ? undefined : 'rotate(-90deg)'} fill="base.500" />
          <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.500">
            {t(`workflows.${category}Workflows`)}
          </Text>
        </Flex>
      </Button>
      <Collapse in={isOpen}>
        {isLoading ? (
          <Flex alignItems="center" justifyContent="center" p={20}>
            <Spinner />
          </Flex>
        ) : data.length ? (
          data.map((workflow) => <WorkflowListItem workflow={workflow} key={workflow.workflow_id} />)
        ) : (
          <IAINoContentFallback
            fontSize="sm"
            py={4}
            label={searchTerm ? t('nodes.noMatchingWorkflows') : t('nodes.noWorkflows')}
            icon={null}
          />
        )}
      </Collapse>
    </Flex>
  );
};

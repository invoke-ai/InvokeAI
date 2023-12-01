import { Divider, Flex, Heading, Spacer } from '@chakra-ui/react';
import {
  IAINoContentFallback,
  IAINoContentFallbackWithSpinner,
} from 'common/components/IAIImageFallback';
import WorkflowLibraryListItem from 'features/workflowLibrary/components/WorkflowLibraryListItem';
import ScrollableContent from 'features/nodes/components/sidePanel/ScrollableContent';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListRecentWorkflowsQuery } from 'services/api/endpoints/workflows';

const WorkflowLibraryRecentList = () => {
  const { t } = useTranslation();
  const { data, isLoading, isError } = useListRecentWorkflowsQuery();

  if (isLoading) {
    return <IAINoContentFallbackWithSpinner label={t('workflows.loading')} />;
  }

  if (!data || isError) {
    return <IAINoContentFallback label={t('workflows.problemLoading')} />;
  }

  if (!data.items.length) {
    return <IAINoContentFallback label={t('workflows.noRecentWorkflows')} />;
  }

  return (
    <>
      <Flex gap={4} alignItems="center" h={10} flexShrink={0} flexGrow={0}>
        <Heading size="md">{t('workflows.recentWorkflows')}</Heading>
        <Spacer />
      </Flex>
      <Divider />
      <ScrollableContent>
        <Flex w="full" h="full" gap={2} flexDir="column">
          {data.items.map((w) => (
            <WorkflowLibraryListItem key={w.workflow_id} workflowDTO={w} />
          ))}
        </Flex>
      </ScrollableContent>
    </>
  );
};

export default memo(WorkflowLibraryRecentList);

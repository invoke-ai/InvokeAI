import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { FormLayout } from 'features/nodes/components/sidePanel/builder/WorkflowBuilder';
import { ViewContextProvider } from 'features/nodes/contexts/ViewContext';
import { selectFormIsEmpty } from 'features/nodes/store/workflowSlice';
import { t } from 'i18next';
import { memo } from 'react';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';

import { EmptyState } from './EmptyState';

export const ViewModeLeftPanelContent = memo(() => {
  return (
    <ViewContextProvider view="view-mode-linear">
      <Box position="relative" w="full" h="full">
        <ScrollableContent>
          <Flex justifyContent="center" w="full" h="full" p={4}>
            <Flex flexDir="column" w="full" h="full" maxW="768px" gap={4}>
              <ViewModeLeftPanelContentInner />
            </Flex>
          </Flex>
        </ScrollableContent>
      </Box>
    </ViewContextProvider>
  );
});
ViewModeLeftPanelContent.displayName = 'ViewModeLeftPanelContent';

const ViewModeLeftPanelContentInner = memo(() => {
  const { isLoading } = useGetOpenAPISchemaQuery();
  const isEmpty = useAppSelector(selectFormIsEmpty);

  if (isLoading) {
    return <IAINoContentFallback label={t('nodes.loadingNodes')} icon={null} />;
  }

  if (isEmpty) {
    return <EmptyState />;
  }

  return <FormLayout />;
});
ViewModeLeftPanelContentInner.displayName = ' ViewModeLeftPanelContentInner';

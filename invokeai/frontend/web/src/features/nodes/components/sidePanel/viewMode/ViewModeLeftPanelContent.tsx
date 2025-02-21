import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { FormElementComponent } from 'features/nodes/components/sidePanel/builder/ContainerElementComponent';
import { EmptyState } from 'features/nodes/components/sidePanel/viewMode/EmptyState';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { selectFormRootElementId, selectIsFormEmpty } from 'features/nodes/store/workflowSlice';
import { t } from 'i18next';
import { memo } from 'react';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';

export const ViewModeLeftPanelContent = memo(() => {
  return (
    <Box position="relative" w="full" h="full">
      <ScrollableContent>
        <ViewModeLeftPanelContentInner />
      </ScrollableContent>
    </Box>
  );
});
ViewModeLeftPanelContent.displayName = 'ViewModeLeftPanelContent';

const ViewModeLeftPanelContentInner = memo(() => {
  const { isLoading } = useGetOpenAPISchemaQuery();
  const loadedTemplates = useStore($hasTemplates);
  const rootElementId = useAppSelector(selectFormRootElementId);
  const isFormEmpty = useAppSelector(selectIsFormEmpty);

  if (isLoading || !loadedTemplates) {
    return <IAINoContentFallback label={t('nodes.loadingNodes')} icon={null} />;
  }

  if (isFormEmpty) {
    return <EmptyState />;
  }

  return (
    <Flex flexDir="column" w="full" maxW="768px">
      <FormElementComponent id={rootElementId} />
    </Flex>
  );
});
ViewModeLeftPanelContentInner.displayName = ' ViewModeLeftPanelContentInner';

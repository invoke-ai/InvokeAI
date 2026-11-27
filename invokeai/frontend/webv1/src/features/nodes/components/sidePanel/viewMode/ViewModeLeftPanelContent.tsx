import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { RootContainerElementViewMode } from 'features/nodes/components/sidePanel/builder/ContainerElement';
import { EmptyState } from 'features/nodes/components/sidePanel/viewMode/EmptyState';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { selectIsFormEmpty } from 'features/nodes/store/selectors';
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
  const isFormEmpty = useAppSelector(selectIsFormEmpty);

  if (isLoading || !loadedTemplates) {
    return <IAINoContentFallback label={t('nodes.loadingNodes')} icon={null} />;
  }

  if (isFormEmpty) {
    return <EmptyState />;
  }

  return (
    <Flex w="full" justifyContent="center">
      <RootContainerElementViewMode />
    </Flex>
  );
});
ViewModeLeftPanelContentInner.displayName = ' ViewModeLeftPanelContentInner';

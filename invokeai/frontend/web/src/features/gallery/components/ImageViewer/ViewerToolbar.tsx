import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import { ToggleProgressButton } from 'features/gallery/components/ImageViewer/ToggleProgressButton';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useMemo } from 'react';

import CurrentImageButtons from './CurrentImageButtons';
import { ViewerToggleMenu } from './ViewerToggleMenu';

export const ViewerToolbar = memo(() => {
  const workflowsMode = useAppSelector((s) => s.workflow.mode);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const shouldShowToggleMenu = useMemo(() => {
    if (activeTabName !== 'workflows') {
      return true;
    }
    return workflowsMode === 'edit';
  }, [workflowsMode, activeTabName]);

  return (
    <Flex w="full" gap={2}>
      <Flex flex={1} justifyContent="center">
        <Flex gap={2} marginInlineEnd="auto">
          <ToggleProgressButton />
          <ToggleMetadataViewerButton />
        </Flex>
      </Flex>
      <Flex flex={1} gap={2} justifyContent="center">
        <CurrentImageButtons />
      </Flex>
      <Flex flex={1} justifyContent="center">
        <Flex gap={2} marginInlineStart="auto">
          {shouldShowToggleMenu && <ViewerToggleMenu />}
        </Flex>
      </Flex>
    </Flex>
  );
});

ViewerToolbar.displayName = 'ViewerToolbar';

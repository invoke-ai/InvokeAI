import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import CurrentImagePreview from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { ImageComparison } from 'features/gallery/components/ImageViewer/ImageComparison';
import { ImageComparisonToolbarButtons } from 'features/gallery/components/ImageViewer/ImageComparisonToolbarButtons';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import { ToggleProgressButton } from 'features/gallery/components/ImageViewer/ToggleProgressButton';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

import CurrentImageButtons from './CurrentImageButtons';
import { ViewerToggleMenu } from './ViewerToggleMenu';

const VIEWER_ENABLED_TABS: InvokeTabName[] = ['canvas', 'generation', 'workflows'];

export const ImageViewer = memo(() => {
  const { viewerMode, onToggle, openEditor } = useImageViewer();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const isViewerEnabled = useMemo(() => VIEWER_ENABLED_TABS.includes(activeTabName), [activeTabName]);
  const shouldShowViewer = useMemo(() => {
    if (!isViewerEnabled) {
      return false;
    }
    return viewerMode === 'view' || viewerMode === 'compare';
  }, [viewerMode, isViewerEnabled]);

  useHotkeys('z', onToggle, { enabled: isViewerEnabled }, [isViewerEnabled, onToggle]);
  useHotkeys('esc', openEditor, { enabled: isViewerEnabled }, [isViewerEnabled, openEditor]);

  if (!shouldShowViewer) {
    return null;
  }

  return (
    <Flex
      layerStyle="first"
      borderRadius="base"
      position="absolute"
      flexDirection="column"
      top={0}
      right={0}
      bottom={0}
      left={0}
      p={2}
      rowGap={4}
      alignItems="center"
      justifyContent="center"
      zIndex={10} // reactflow puts its minimap at 5, so we need to be above that
    >
      <Flex w="full" gap={2}>
        <Flex flex={1} justifyContent="center">
          <Flex gap={2} marginInlineEnd="auto">
            <ToggleProgressButton />
            <ToggleMetadataViewerButton />
          </Flex>
        </Flex>
        <Flex flex={1} gap={2} justifyContent="center">
          {viewerMode === 'view' && <CurrentImageButtons />}
          {viewerMode === 'compare' && <ImageComparisonToolbarButtons />}
        </Flex>
        <Flex flex={1} justifyContent="center">
          <Flex gap={2} marginInlineStart="auto">
            <ViewerToggleMenu />
          </Flex>
        </Flex>
      </Flex>
      <Box w="full" h="full">
        {viewerMode === 'view' && <CurrentImagePreview />}
        {viewerMode === 'compare' && <ImageComparison />}
      </Box>
    </Flex>
  );
});

ImageViewer.displayName = 'ImageViewer';

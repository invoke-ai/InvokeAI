import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CompareToolbar } from 'features/gallery/components/ImageViewer/CompareToolbar';
import CurrentImagePreview from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { ImageComparison } from 'features/gallery/components/ImageViewer/ImageComparison';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { ViewerToolbar } from 'features/gallery/components/ImageViewer/ViewerToolbar';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

const VIEWER_ENABLED_TABS: InvokeTabName[] = ['canvas', 'generation', 'workflows'];

export const ImageViewer = memo(() => {
  const { isOpen, onToggle, onClose } = useImageViewer();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const workflowsMode = useAppSelector((s) => s.workflow.mode);
  const isComparing = useAppSelector((s) => s.gallery.imageToCompare !== null);
  const isViewerEnabled = useMemo(() => VIEWER_ENABLED_TABS.includes(activeTabName), [activeTabName]);
  const shouldShowViewer = useMemo(() => {
    if (activeTabName === 'workflows' && workflowsMode === 'view') {
      return true;
    }
    if (!isViewerEnabled) {
      return false;
    }
    return isOpen;
  }, [isOpen, isViewerEnabled, workflowsMode, activeTabName]);

  useHotkeys('z', onToggle, { enabled: isViewerEnabled }, [isViewerEnabled, onToggle]);
  useHotkeys('esc', onClose, { enabled: isViewerEnabled }, [isViewerEnabled, onClose]);

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
    >
      {isComparing && <CompareToolbar />}
      {!isComparing && <ViewerToolbar />}
      <Box w="full" h="full">
        {!isComparing && <CurrentImagePreview />}
        {isComparing && <ImageComparison />}
      </Box>
    </Flex>
  );
});

ImageViewer.displayName = 'ImageViewer';

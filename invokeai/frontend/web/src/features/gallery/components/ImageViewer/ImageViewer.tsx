import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import { ToggleProgressButton } from 'features/gallery/components/ImageViewer/ToggleProgressButton';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import type { AnimationProps } from 'framer-motion';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

import CurrentImageButtons from './CurrentImageButtons';
import CurrentImagePreview from './CurrentImagePreview';
import { EditorButton } from './EditorButton';

const initial: AnimationProps['initial'] = {
  opacity: 0,
};
const animate: AnimationProps['animate'] = {
  opacity: 1,
  transition: { duration: 0.07 },
};
const exit: AnimationProps['exit'] = {
  opacity: 0,
  transition: { duration: 0.07 },
};

const VIEWER_ENABLED_TABS: InvokeTabName[] = ['canvas', 'generation', 'workflows'];

export const ImageViewer = memo(() => {
  const { isOpen, onToggle, onClose } = useImageViewer();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const isViewerEnabled = useMemo(() => VIEWER_ENABLED_TABS.includes(activeTabName), [activeTabName]);
  const shouldShowViewer = useMemo(() => {
    if (!isViewerEnabled) {
      return false;
    }
    return isOpen;
  }, [isOpen, isViewerEnabled]);

  useHotkeys('z', onToggle, { enabled: isViewerEnabled }, [isViewerEnabled, onToggle]);
  useHotkeys('esc', onClose, { enabled: isViewerEnabled }, [isViewerEnabled, onClose]);

  // The AnimatePresence mode must be wait - else framer can get confused if you spam the toggle button
  return (
    <AnimatePresence mode="wait">
      {shouldShowViewer && (
        <Flex
          key="imageViewer"
          as={motion.div}
          initial={initial}
          animate={animate}
          exit={exit}
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
              <CurrentImageButtons />
            </Flex>
            <Flex flex={1} justifyContent="center">
              <Flex gap={2} marginInlineStart="auto">
                <EditorButton />
              </Flex>
            </Flex>
          </Flex>
          <CurrentImagePreview />
        </Flex>
      )}
    </AnimatePresence>
  );
});

ImageViewer.displayName = 'ImageViewer';

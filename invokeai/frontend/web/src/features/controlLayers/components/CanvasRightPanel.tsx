import { useDndContext } from '@dnd-kit/core';
import { Box, Button, Spacer, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useScopeOnFocus } from 'common/hooks/interactionScopes';
import { CanvasLayersPanelContent } from 'features/controlLayers/components/CanvasLayersPanelContent';
import {
  $canvasRightPanelTabIndex,
  selectCanvasRightPanelGalleryTab,
  selectCanvasRightPanelLayersTab,
} from 'features/controlLayers/store/ephemeral';
import { selectEntityCountActive } from 'features/controlLayers/store/selectors';
import GalleryPanelContent from 'features/gallery/components/GalleryPanelContent';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

export const CanvasRightPanel = memo(() => {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const tabIndex = useStore($canvasRightPanelTabIndex);
  useScopeOnFocus('gallery', ref);
  const imageViewer = useImageViewer();
  const onClickViewerToggleButton = useCallback(() => {
    if ($canvasRightPanelTabIndex.get() !== 1) {
      $canvasRightPanelTabIndex.set(1);
    }
    imageViewer.toggle();
  }, [imageViewer]);
  useHotkeys('z', imageViewer.toggle);

  return (
    <Tabs index={tabIndex} onChange={$canvasRightPanelTabIndex.set} w="full" h="full" display="flex" flexDir="column">
      <TabList alignItems="center">
        <PanelTabs />
        <Spacer />
        <Button size="sm" variant="ghost" onClick={onClickViewerToggleButton}>
          {imageViewer.isOpen ? t('gallery.closeViewer') : t('gallery.openViewer')}
        </Button>
      </TabList>
      <TabPanels w="full" h="full">
        <TabPanel w="full" h="full" p={0} pt={3}>
          <CanvasLayersPanelContent />
        </TabPanel>
        <TabPanel w="full" h="full" p={0} pt={3}>
          <GalleryPanelContent />
        </TabPanel>
      </TabPanels>
    </Tabs>
  );
});

CanvasRightPanel.displayName = 'CanvasRightPanel';

const PanelTabs = memo(() => {
  const { t } = useTranslation();
  const activeEntityCount = useAppSelector(selectEntityCountActive);
  const tabTimeout = useRef<number | null>(null);
  const dndCtx = useDndContext();

  const onOnMouseOverLayersTab = useCallback(() => {
    tabTimeout.current = window.setTimeout(() => {
      if (dndCtx.active) {
        selectCanvasRightPanelLayersTab();
      }
    }, 300);
  }, [dndCtx.active]);

  const onOnMouseOverGalleryTab = useCallback(() => {
    tabTimeout.current = window.setTimeout(() => {
      if (dndCtx.active) {
        selectCanvasRightPanelGalleryTab();
      }
    }, 300);
  }, [dndCtx.active]);

  const onMouseOut = useCallback(() => {
    if (tabTimeout.current) {
      clearTimeout(tabTimeout.current);
    }
  }, []);

  const layersTabLabel = useMemo(() => {
    if (activeEntityCount === 0) {
      return t('controlLayers.layer_other');
    }
    return `${t('controlLayers.layer_other')} (${activeEntityCount})`;
  }, [activeEntityCount, t]);

  return (
    <>
      <Tab position="relative" onMouseOver={onOnMouseOverLayersTab} onMouseOut={onMouseOut} w={32}>
        <Box as="span" w="full">
          {layersTabLabel}
        </Box>
      </Tab>
      <Tab position="relative" onMouseOver={onOnMouseOverGalleryTab} onMouseOut={onMouseOut}>
        {t('gallery.gallery')}
      </Tab>
    </>
  );
});

PanelTabs.displayName = 'PanelTabs';

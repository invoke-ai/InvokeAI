import { useDndContext } from '@dnd-kit/core';
import { Box, Button, Spacer, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import { CanvasLayersPanelContent } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectEntityCountActive } from 'features/controlLayers/store/selectors';
import type { DragToLayersTabDropData } from 'features/dnd/types';
import GalleryPanelContent from 'features/gallery/components/GalleryPanelContent';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { selectActiveTabCanvasRightPanel } from 'features/ui/store/uiSelectors';
import { activeTabCanvasRightPanelChanged } from 'features/ui/store/uiSlice';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasRightPanel = memo(() => {
  const { t } = useTranslation();
  const activeTab = useAppSelector(selectActiveTabCanvasRightPanel);
  const imageViewer = useImageViewer();
  const dispatch = useAppDispatch();

  const tabIndex = useMemo(() => {
    if (activeTab === 'gallery') {
      return 1;
    } else {
      return 0;
    }
  }, [activeTab]);

  const onClickViewerToggleButton = useCallback(() => {
    if (activeTab !== 'gallery') {
      dispatch(activeTabCanvasRightPanelChanged('gallery'));
    }
    imageViewer.toggle();
  }, [imageViewer, activeTab, dispatch]);

  const onChangeTab = useCallback(
    (index: number) => {
      if (index === 0) {
        dispatch(activeTabCanvasRightPanelChanged('layers'));
      } else {
        dispatch(activeTabCanvasRightPanelChanged('gallery'));
      }
    },
    [dispatch]
  );

  useRegisteredHotkeys({
    id: 'toggleViewer',
    category: 'viewer',
    callback: imageViewer.toggle,
    dependencies: [imageViewer],
  });

  return (
    <Tabs index={tabIndex} onChange={onChangeTab} w="full" h="full" display="flex" flexDir="column">
      <TabList alignItems="center">
        <PanelTabs />
        <Spacer />
        <Button size="sm" variant="ghost" onClick={onClickViewerToggleButton}>
          {imageViewer.isOpen ? t('gallery.closeViewer') : t('gallery.openViewer')}
        </Button>
      </TabList>
      <TabPanels w="full" h="full">
        <TabPanel w="full" h="full" p={0} pt={3}>
          <CanvasManagerProviderGate>
            <CanvasLayersPanelContent />
          </CanvasManagerProviderGate>
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
  const activeTab = useAppSelector(selectActiveTabCanvasRightPanel);
  const activeEntityCount = useAppSelector(selectEntityCountActive);
  const tabTimeout = useRef<number | null>(null);
  const dndCtx = useDndContext();
  const dispatch = useAppDispatch();

  const droppableData: DragToLayersTabDropData = useMemo(
    () => ({
      id: 'DRAG_TO_LAYERS_TAB',
      actionType: 'DRAG_TO_LAYERS_TAB',
    }),
    []
  );

  const onOnMouseOverLayersTab = useCallback(() => {
    tabTimeout.current = window.setTimeout(() => {
      if (dndCtx.active) {
        dispatch(activeTabCanvasRightPanelChanged('layers'));
      }
    }, 300);
  }, [dndCtx.active, dispatch]);

  const onOnMouseOverGalleryTab = useCallback(() => {
    tabTimeout.current = window.setTimeout(() => {
      if (dndCtx.active) {
        dispatch(activeTabCanvasRightPanelChanged('gallery'));
      }
    }, 300);
  }, [dndCtx.active, dispatch]);

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
        {activeTab === 'gallery' && (
          <IAIDroppable
            data={droppableData}
            dropLabel={t('controlLayers.replaceLayer')}
            textStyleOverrides={{ fontSize: 'sm' }}
          />
        )}
      </Tab>
      <Tab position="relative" onMouseOver={onOnMouseOverGalleryTab} onMouseOut={onMouseOut}>
        {t('gallery.gallery')}
      </Tab>
    </>
  );
});

PanelTabs.displayName = 'PanelTabs';

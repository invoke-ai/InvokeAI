import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { dropTargetForElements, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { dropTargetForExternal, monitorForExternal } from '@atlaskit/pragmatic-drag-and-drop/external/adapter';
import { Box, Button, Spacer, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import { CanvasLayersPanelContent } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectEntityCountActive } from 'features/controlLayers/store/selectors';
import { DndDropOverlay } from 'features/dnd2/DndDropOverlay';
import type { DndState } from 'features/dnd2/types';
import GalleryPanelContent from 'features/gallery/components/GalleryPanelContent';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { selectActiveTabCanvasRightPanel } from 'features/ui/store/uiSelectors';
import { activeTabCanvasRightPanelChanged } from 'features/ui/store/uiSlice';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
  const store = useAppStore();
  const activeEntityCount = useAppSelector(selectEntityCountActive);
  const [layersTabDndState, setLayersTabDndState] = useState<DndState>('idle');
  const [galleryTabDndState, setGalleryTabDndState] = useState<DndState>('idle');
  const layersTabRef = useRef<HTMLDivElement>(null);
  const galleryTabRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<number | null>(null);

  const layersTabLabel = useMemo(() => {
    if (activeEntityCount === 0) {
      return t('controlLayers.layer_other');
    }
    return `${t('controlLayers.layer_other')} (${activeEntityCount})`;
  }, [activeEntityCount, t]);

  useEffect(() => {
    if (!layersTabRef.current) {
      return;
    }

    const onDragEnter = () => {
      // If we are already on the layers tab, do nothing
      if (selectActiveTabCanvasRightPanel(store.getState()) === 'layers') {
        return;
      }

      // Else set the state to active and switch to the layers tab after a timeout
      setLayersTabDndState('over');
      timeoutRef.current = window.setTimeout(() => {
        timeoutRef.current = null;
        store.dispatch(activeTabCanvasRightPanelChanged('layers'));
        // When we switch tabs, the other tab should be pending
        setLayersTabDndState('idle');
        setGalleryTabDndState('potential');
      }, 300);
    };
    const onDragLeave = () => {
      // Set the state to idle or pending depending on the current tab
      if (selectActiveTabCanvasRightPanel(store.getState()) === 'layers') {
        setLayersTabDndState('idle');
      } else {
        setLayersTabDndState('potential');
      }
      // Abort the tab switch if it hasn't happened yet
      if (timeoutRef.current !== null) {
        clearTimeout(timeoutRef.current);
      }
    };
    const canMonitor = () => {
      // Only monitor if we are not already on the layers tab
      return selectActiveTabCanvasRightPanel(store.getState()) !== 'layers';
    };
    const onDragStart = () => {
      // Set the state to pending when a drag starts
      setLayersTabDndState('potential');
    };
    return combine(
      dropTargetForElements({
        element: layersTabRef.current,
        onDragEnter,
        onDragLeave,
      }),
      monitorForElements({
        canMonitor,
        onDragStart,
      }),
      dropTargetForExternal({
        element: layersTabRef.current,
        onDragEnter,
        onDragLeave,
      }),
      monitorForExternal({
        canMonitor,
        onDragStart,
      })
    );
  }, [store]);

  useEffect(() => {
    if (!galleryTabRef.current) {
      return;
    }

    const onDragEnter = () => {
      // If we are already on the gallery tab, do nothing
      if (selectActiveTabCanvasRightPanel(store.getState()) === 'gallery') {
        return;
      }

      // Else set the state to active and switch to the gallery tab after a timeout
      setGalleryTabDndState('over');
      timeoutRef.current = window.setTimeout(() => {
        timeoutRef.current = null;
        store.dispatch(activeTabCanvasRightPanelChanged('gallery'));
        // When we switch tabs, the other tab should be pending
        setGalleryTabDndState('idle');
        setLayersTabDndState('potential');
      }, 300);
    };

    const onDragLeave = () => {
      // Set the state to idle or pending depending on the current tab
      if (selectActiveTabCanvasRightPanel(store.getState()) === 'gallery') {
        setGalleryTabDndState('idle');
      } else {
        setGalleryTabDndState('potential');
      }
      // Abort the tab switch if it hasn't happened yet
      if (timeoutRef.current !== null) {
        clearTimeout(timeoutRef.current);
      }
    };

    const canMonitor = () => {
      // Only monitor if we are not already on the gallery tab
      return selectActiveTabCanvasRightPanel(store.getState()) !== 'gallery';
    };

    const onDragStart = () => {
      // Set the state to pending when a drag starts
      setGalleryTabDndState('potential');
    };

    return combine(
      dropTargetForElements({
        element: galleryTabRef.current,
        onDragEnter,
        onDragLeave,
      }),
      monitorForElements({
        canMonitor,
        onDragStart,
      }),
      dropTargetForExternal({
        element: galleryTabRef.current,
        onDragEnter,
        onDragLeave,
      }),
      monitorForExternal({
        canMonitor,
        onDragStart,
      })
    );
  }, [store]);

  useEffect(() => {
    const onDrop = () => {
      // Reset the dnd state when a drop happens
      setGalleryTabDndState('idle');
      setLayersTabDndState('idle');
    };
    const cleanup = combine(monitorForElements({ onDrop }), monitorForExternal({ onDrop }));

    return () => {
      cleanup();
      if (timeoutRef.current !== null) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (
    <>
      <Tab ref={layersTabRef} position="relative" w={32}>
        <Box as="span" w="full">
          {layersTabLabel}
        </Box>
        <DndDropOverlay dndState={layersTabDndState} withBackdrop={false} />
      </Tab>
      <Tab ref={galleryTabRef} position="relative" w={32}>
        <Box as="span" w="full">
          {t('gallery.gallery')}
        </Box>
        <DndDropOverlay dndState={galleryTabDndState} withBackdrop={false} />
      </Tab>
    </>
  );
});

PanelTabs.displayName = 'PanelTabs';

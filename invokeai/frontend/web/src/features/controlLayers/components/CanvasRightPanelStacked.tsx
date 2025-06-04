import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { dropTargetForElements, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { dropTargetForExternal, monitorForExternal } from '@atlaskit/pragmatic-drag-and-drop/external/adapter';
import { Box, Tab } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import { CanvasLayersPanelContent } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectEntityCountActive } from 'features/controlLayers/store/selectors';
import { multipleImageDndSource, singleImageDndSource } from 'features/dnd/dnd';
import { DndDropOverlay } from 'features/dnd/DndDropOverlay';
import type { DndTargetState } from 'features/dnd/types';
import RightPanelContent from 'features/gallery/components/GalleryTopBar';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { selectActiveTabCanvasRightPanel } from 'features/ui/store/uiSelectors';
import { activeTabCanvasRightPanelChanged } from 'features/ui/store/uiSlice';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';

export const CanvasRightPanelStacked = memo(() => {
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
    imageViewer.open();
  }, [imageViewer]);

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

  return (
    <PanelGroup direction="vertical">
      <Panel>
        <RightPanelContent />
      </Panel>
      <PanelResizeHandle />
      <Panel>
        <CanvasManagerProviderGate>
          <CanvasLayersPanelContent />
        </CanvasManagerProviderGate>
      </Panel>
    </PanelGroup>
  );
});

CanvasRightPanelStacked.displayName = 'CanvasRightPanelStacked';

const PanelTabs = memo(() => {
  const { t } = useTranslation();
  const store = useAppStore();
  const activeEntityCount = useAppSelector(selectEntityCountActive);
  const [layersTabDndState, setLayersTabDndState] = useState<DndTargetState>('idle');
  const [galleryTabDndState, setGalleryTabDndState] = useState<DndTargetState>('idle');
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

    const getIsOnLayersTab = () => selectActiveTabCanvasRightPanel(store.getState()) === 'layers';

    const onDragEnter = () => {
      // If we are already on the layers tab, do nothing
      if (getIsOnLayersTab()) {
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
      if (getIsOnLayersTab()) {
        setLayersTabDndState('idle');
      } else {
        setLayersTabDndState('potential');
      }
      // Abort the tab switch if it hasn't happened yet
      if (timeoutRef.current !== null) {
        clearTimeout(timeoutRef.current);
      }
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
        canMonitor: ({ source }) => {
          if (!singleImageDndSource.typeGuard(source.data) && !multipleImageDndSource.typeGuard(source.data)) {
            return false;
          }
          // Only monitor if we are not already on the gallery tab
          return !getIsOnLayersTab();
        },
        onDragStart,
      }),
      dropTargetForExternal({
        element: layersTabRef.current,
        onDragEnter,
        onDragLeave,
      }),
      monitorForExternal({
        canMonitor: () => !getIsOnLayersTab(),
        onDragStart,
      })
    );
  }, [store]);

  useEffect(() => {
    if (!galleryTabRef.current) {
      return;
    }

    const getIsOnGalleryTab = () => selectActiveTabCanvasRightPanel(store.getState()) === 'gallery';

    const onDragEnter = () => {
      // If we are already on the gallery tab, do nothing
      if (getIsOnGalleryTab()) {
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
      if (getIsOnGalleryTab()) {
        setGalleryTabDndState('idle');
      } else {
        setGalleryTabDndState('potential');
      }
      // Abort the tab switch if it hasn't happened yet
      if (timeoutRef.current !== null) {
        clearTimeout(timeoutRef.current);
      }
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
        canMonitor: ({ source }) => {
          if (!singleImageDndSource.typeGuard(source.data) && !multipleImageDndSource.typeGuard(source.data)) {
            return false;
          }
          // Only monitor if we are not already on the gallery tab
          return !getIsOnGalleryTab();
        },
        onDragStart,
      }),
      dropTargetForExternal({
        element: galleryTabRef.current,
        onDragEnter,
        onDragLeave,
      }),
      monitorForExternal({
        canMonitor: () => !getIsOnGalleryTab(),
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

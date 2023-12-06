import {
  Icon,
  Spacer,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  Tooltip,
  VisuallyHidden,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ImageGalleryContent from 'features/gallery/components/ImageGalleryContent';
import NodeEditorPanelGroup from 'features/nodes/components/sidePanel/NodeEditorPanelGroup';
import { usePanel } from 'features/ui/hooks/usePanel';
import { usePanelStorage } from 'features/ui/hooks/usePanelStorage';
import { InvokeTabName } from 'features/ui/store/tabMap';
import {
  activeTabIndexSelector,
  activeTabNameSelector,
} from 'features/ui/store/uiSelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { ResourceKey } from 'i18next';
import { isEqual } from 'lodash-es';
import { MouseEvent, ReactNode, memo, useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaCube, FaFont, FaImage, FaStream } from 'react-icons/fa';
import { FaCircleNodes } from 'react-icons/fa6';
import { MdGridOn } from 'react-icons/md';
import { Panel, PanelGroup } from 'react-resizable-panels';
import FloatingGalleryButton from './FloatingGalleryButton';
import FloatingSidePanelButtons from './FloatingParametersPanelButtons';
import ParametersPanel from './ParametersPanel';
import ImageTab from './tabs/ImageToImage/ImageToImageTab';
import ModelManagerTab from './tabs/ModelManager/ModelManagerTab';
import NodesTab from './tabs/Nodes/NodesTab';
import QueueTab from './tabs/Queue/QueueTab';
import ResizeHandle from './tabs/ResizeHandle';
import TextToImageTab from './tabs/TextToImage/TextToImageTab';
import UnifiedCanvasTab from './tabs/UnifiedCanvas/UnifiedCanvasTab';

export interface InvokeTabInfo {
  id: InvokeTabName;
  translationKey: string;
  icon: ReactNode;
  content: ReactNode;
}

const tabs: InvokeTabInfo[] = [
  {
    id: 'txt2img',
    translationKey: 'common.txt2img',
    icon: <Icon as={FaFont} sx={{ boxSize: 6, pointerEvents: 'none' }} />,
    content: <TextToImageTab />,
  },
  {
    id: 'img2img',
    translationKey: 'common.img2img',
    icon: <Icon as={FaImage} sx={{ boxSize: 6, pointerEvents: 'none' }} />,
    content: <ImageTab />,
  },
  {
    id: 'unifiedCanvas',
    translationKey: 'common.unifiedCanvas',
    icon: <Icon as={MdGridOn} sx={{ boxSize: 6, pointerEvents: 'none' }} />,
    content: <UnifiedCanvasTab />,
  },
  {
    id: 'nodes',
    translationKey: 'common.nodes',
    icon: (
      <Icon as={FaCircleNodes} sx={{ boxSize: 6, pointerEvents: 'none' }} />
    ),
    content: <NodesTab />,
  },
  {
    id: 'modelManager',
    translationKey: 'modelManager.modelManager',
    icon: <Icon as={FaCube} sx={{ boxSize: 6, pointerEvents: 'none' }} />,
    content: <ModelManagerTab />,
  },
  {
    id: 'queue',
    translationKey: 'queue.queue',
    icon: <Icon as={FaStream} sx={{ boxSize: 6, pointerEvents: 'none' }} />,
    content: <QueueTab />,
  },
];

const enabledTabsSelector = createSelector(
  [stateSelector],
  ({ config }) => {
    const { disabledTabs } = config;
    const enabledTabs = tabs.filter((tab) => !disabledTabs.includes(tab.id));
    return enabledTabs;
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
  }
);

const SIDE_PANEL_MIN_SIZE_PX = 448;
const MAIN_PANEL_MIN_SIZE_PX = 448;
const GALLERY_PANEL_MIN_SIZE_PX = 360;

export const NO_GALLERY_TABS: InvokeTabName[] = ['modelManager', 'queue'];
export const NO_SIDE_PANEL_TABS: InvokeTabName[] = ['modelManager', 'queue'];

const InvokeTabs = () => {
  const activeTabIndex = useAppSelector(activeTabIndexSelector);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const enabledTabs = useAppSelector(enabledTabsSelector);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const handleClickTab = useCallback((e: MouseEvent<HTMLElement>) => {
    if (e.target instanceof HTMLElement) {
      e.target.blur();
    }
  }, []);

  const tabs = useMemo(
    () =>
      enabledTabs.map((tab) => (
        <Tooltip
          key={tab.id}
          hasArrow
          label={String(t(tab.translationKey as ResourceKey))}
          placement="end"
        >
          <Tab onClick={handleClickTab}>
            <VisuallyHidden>
              {String(t(tab.translationKey as ResourceKey))}
            </VisuallyHidden>
            {tab.icon}
          </Tab>
        </Tooltip>
      )),
    [enabledTabs, t, handleClickTab]
  );

  const tabPanels = useMemo(
    () =>
      enabledTabs.map((tab) => <TabPanel key={tab.id}>{tab.content}</TabPanel>),
    [enabledTabs]
  );

  const handleTabChange = useCallback(
    (index: number) => {
      const tab = enabledTabs[index];
      if (!tab) {
        return;
      }
      dispatch(setActiveTab(tab.id));
    },
    [dispatch, enabledTabs]
  );

  const {
    ref: sidePanelRef,
    reset: resetSidePanel,
    expand: expandSidePanel,
    collapse: collapseSidePanel,
    toggle: toggleSidePanel,
  } = usePanel({ sizePixels: SIDE_PANEL_MIN_SIZE_PX });

  const {
    ref: galleryPanelRef,
    reset: resetGalleryPanel,
    expand: expandGalleryPanel,
    collapse: collapseGalleryPanel,
    toggle: toggleGalleryPanel,
  } = usePanel({ sizePixels: GALLERY_PANEL_MIN_SIZE_PX });

  useHotkeys(
    'f',
    () => {
      if (
        sidePanelRef.current?.isCollapsed() ||
        galleryPanelRef.current?.isCollapsed()
      ) {
        expandGalleryPanel();
        expandSidePanel();
      } else {
        collapseSidePanel();
        collapseGalleryPanel();
      }
    },
    [dispatch, sidePanelRef, galleryPanelRef]
  );

  useHotkeys(
    ['t', 'o'],
    () => {
      toggleSidePanel();
    },
    [dispatch]
  );

  useHotkeys(
    'g',
    () => {
      toggleGalleryPanel();
    },
    [dispatch]
  );

  const panelStorage = usePanelStorage();

  console.log({
    sidePanelRef,
    resetSidePanel,
    expandSidePanel,
    collapseSidePanel,
    toggleSidePanel,
  });

  console.log({
    galleryPanelRef,
    resetGalleryPanel,
    expandGalleryPanel,
    collapseGalleryPanel,
    toggleGalleryPanel,
  });

  return (
    <Tabs
      variant="appTabs"
      defaultIndex={activeTabIndex}
      index={activeTabIndex}
      onChange={handleTabChange}
      sx={{
        flexGrow: 1,
        gap: 4,
      }}
      isLazy
    >
      <TabList
        sx={{
          pt: 2,
          gap: 4,
          flexDir: 'column',
        }}
      >
        {tabs}
        <Spacer />
      </TabList>
      <PanelGroup
        id="app"
        autoSaveId="app"
        direction="horizontal"
        style={{ height: '100%', width: '100%' }}
        storage={panelStorage}
      >
        {!NO_SIDE_PANEL_TABS.includes(activeTabName) && (
          <>
            <Panel
              order={0}
              id="side"
              ref={sidePanelRef}
              defaultSizePixels={SIDE_PANEL_MIN_SIZE_PX}
              minSizePixels={SIDE_PANEL_MIN_SIZE_PX}
              collapsible
            >
              {activeTabName === 'nodes' ? (
                <NodeEditorPanelGroup />
              ) : (
                <ParametersPanel />
              )}
            </Panel>
            <ResizeHandle
              onDoubleClick={resetSidePanel}
              collapsedDirection={
                sidePanelRef.current?.isCollapsed() ? 'left' : undefined
              }
            />
            <FloatingSidePanelButtons
              isSidePanelCollapsed={Boolean(
                sidePanelRef.current?.isCollapsed()
              )}
              sidePanelRef={sidePanelRef}
            />
          </>
        )}
        <Panel id="main" order={1} minSizePixels={MAIN_PANEL_MIN_SIZE_PX}>
          <TabPanels style={{ height: '100%', width: '100%' }}>
            {tabPanels}
          </TabPanels>
        </Panel>
        {!NO_GALLERY_TABS.includes(activeTabName) && (
          <>
            <ResizeHandle
              onDoubleClick={resetGalleryPanel}
              collapsedDirection={
                galleryPanelRef.current?.isCollapsed() ? 'right' : undefined
              }
            />
            <Panel
              id="gallery"
              ref={galleryPanelRef}
              order={2}
              defaultSizePixels={GALLERY_PANEL_MIN_SIZE_PX}
              minSizePixels={GALLERY_PANEL_MIN_SIZE_PX}
              collapsible
            >
              <ImageGalleryContent />
            </Panel>
            <FloatingGalleryButton
              isGalleryCollapsed={Boolean(
                galleryPanelRef.current?.isCollapsed()
              )}
              galleryPanelRef={galleryPanelRef}
            />
          </>
        )}
      </PanelGroup>
    </Tabs>
  );
};

export default memo(InvokeTabs);

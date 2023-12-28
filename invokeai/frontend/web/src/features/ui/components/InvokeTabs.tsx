import { Flex, Spacer } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvTab } from 'common/components/InvTabs/InvTab';
import {
  InvTabList,
  InvTabPanel,
  InvTabPanels,
  InvTabs,
} from 'common/components/InvTabs/wrapper';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import ImageGalleryContent from 'features/gallery/components/ImageGalleryContent';
import NodeEditorPanelGroup from 'features/nodes/components/sidePanel/NodeEditorPanelGroup';
import { usePanel } from 'features/ui/hooks/usePanel';
import { usePanelStorage } from 'features/ui/hooks/usePanelStorage';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import {
  activeTabIndexSelector,
  activeTabNameSelector,
} from 'features/ui/store/uiSelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import type { MouseEvent, ReactElement, ReactNode } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaCube, FaFont, FaImage, FaStream } from 'react-icons/fa';
import { FaCircleNodes } from 'react-icons/fa6';
import { MdGridOn } from 'react-icons/md';
import { Panel, PanelGroup } from 'react-resizable-panels';

import FloatingGalleryButton from './FloatingGalleryButton';
import ParametersPanel from './ParametersPanel';
import ImageTab from './tabs/ImageToImageTab';
import ModelManagerTab from './tabs/ModelManagerTab';
import NodesTab from './tabs/NodesTab';
import QueueTab from './tabs/QueueTab';
import ResizeHandle from './tabs/ResizeHandle';
import TextToImageTab from './tabs/TextToImageTab';
import UnifiedCanvasTab from './tabs/UnifiedCanvasTab';

export interface InvokeTabInfo {
  id: InvokeTabName;
  translationKey: string;
  icon: ReactElement;
  content: ReactNode;
}

const tabs: InvokeTabInfo[] = [
  {
    id: 'txt2img',
    translationKey: 'common.txt2img',
    icon: <FaFont />,
    content: <TextToImageTab />,
  },
  {
    id: 'img2img',
    translationKey: 'common.img2img',
    icon: <FaImage />,
    content: <ImageTab />,
  },
  {
    id: 'unifiedCanvas',
    translationKey: 'common.unifiedCanvas',
    icon: <MdGridOn />,
    content: <UnifiedCanvasTab />,
  },
  {
    id: 'nodes',
    translationKey: 'common.nodes',
    icon: <FaCircleNodes />,
    content: <NodesTab />,
  },
  {
    id: 'modelManager',
    translationKey: 'modelManager.modelManager',
    icon: <FaCube />,
    content: <ModelManagerTab />,
  },
  {
    id: 'queue',
    translationKey: 'queue.queue',
    icon: <FaStream />,
    content: <QueueTab />,
  },
];

const enabledTabsSelector = createMemoizedSelector(
  [stateSelector],
  ({ config }) => {
    const { disabledTabs } = config;
    const enabledTabs = tabs.filter((tab) => !disabledTabs.includes(tab.id));
    return enabledTabs;
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
        <InvTooltip key={tab.id} label={t(tab.translationKey)} placement="end">
          <InvTab
            as={InvIconButton}
            p={0}
            onClick={handleClickTab}
            icon={tab.icon}
            size="md"
            fontSize="24px"
            variant="appTab"
            data-selected={activeTabName === tab.id}
            aria-label={t(tab.translationKey)}
          ></InvTab>
        </InvTooltip>
      )),
    [enabledTabs, t, handleClickTab, activeTabName]
  );

  const tabPanels = useMemo(
    () =>
      enabledTabs.map((tab) => (
        <InvTabPanel key={tab.id}>{tab.content}</InvTabPanel>
      )),
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
    isCollapsed: isSidePanelCollapsed,
    expand: expandSidePanel,
    collapse: collapseSidePanel,
    toggle: toggleSidePanel,
  } = usePanel(SIDE_PANEL_MIN_SIZE_PX, 'pixels');

  const {
    ref: galleryPanelRef,
    minSize: galleryPanelMinSize,
    isCollapsed: isGalleryPanelCollapsed,
    setIsCollapsed: setIsGalleryPanelCollapsed,
    reset: resetGalleryPanel,
    expand: expandGalleryPanel,
    collapse: collapseGalleryPanel,
    toggle: toggleGalleryPanel,
  } = usePanel(GALLERY_PANEL_MIN_SIZE_PX, 'pixels');

  useHotkeys(
    'f',
    () => {
      if (isGalleryPanelCollapsed || isSidePanelCollapsed) {
        expandGalleryPanel();
        expandSidePanel();
      } else {
        collapseSidePanel();
        collapseGalleryPanel();
      }
    },
    [dispatch, isGalleryPanelCollapsed, isSidePanelCollapsed]
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

  return (
    <InvTabs
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
      <InvTabList
        sx={{
          gap: 4,
          flexDir: 'column',
        }}
      >
        {tabs}
        <Spacer />
      </InvTabList>
      {!NO_SIDE_PANEL_TABS.includes(activeTabName) && (
        <Flex h="full" w={434} flexShrink={0}>
          {activeTabName === 'nodes' ? (
            <NodeEditorPanelGroup />
          ) : (
            <ParametersPanel />
          )}
        </Flex>
      )}
      <PanelGroup
        id="app"
        autoSaveId="app"
        direction="horizontal"
        style={{ height: '100%', width: '100%' }}
        storage={panelStorage}
        units="pixels"
      >
        {/* {!NO_SIDE_PANEL_TABS.includes(activeTabName) && (
          <>
            <Panel
              order={0}
              id="side"
              ref={sidePanelRef}
              defaultSize={sidePanelMinSize}
              minSize={sidePanelMinSize}
              onCollapse={setIsSidePanelCollapsed}
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
              collapsedDirection={isSidePanelCollapsed ? 'left' : undefined}
            />
            <FloatingSidePanelButtons
              isSidePanelCollapsed={isSidePanelCollapsed}
              sidePanelRef={sidePanelRef}
            />
          </>
        )} */}
        <Panel id="main" order={1} minSize={MAIN_PANEL_MIN_SIZE_PX}>
          <InvTabPanels style={{ height: '100%', width: '100%' }}>
            {tabPanels}
          </InvTabPanels>
        </Panel>
        {!NO_GALLERY_TABS.includes(activeTabName) && (
          <>
            <ResizeHandle
              onDoubleClick={resetGalleryPanel}
              collapsedDirection={isGalleryPanelCollapsed ? 'right' : undefined}
            />
            <Panel
              id="gallery"
              ref={galleryPanelRef}
              order={2}
              defaultSize={galleryPanelMinSize}
              minSize={galleryPanelMinSize}
              onCollapse={setIsGalleryPanelCollapsed}
              collapsible
            >
              <ImageGalleryContent />
            </Panel>
            <FloatingGalleryButton
              isGalleryCollapsed={isGalleryPanelCollapsed}
              galleryPanelRef={galleryPanelRef}
            />
          </>
        )}
      </PanelGroup>
    </InvTabs>
  );
};

export default memo(InvokeTabs);

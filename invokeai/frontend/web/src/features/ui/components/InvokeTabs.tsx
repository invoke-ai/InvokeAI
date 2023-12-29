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
import type { CSSProperties, MouseEvent, ReactElement, ReactNode } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaCube, FaFont, FaImage } from 'react-icons/fa';
import { FaCircleNodes, FaList } from 'react-icons/fa6';
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
    icon: <FaList />,
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

export const NO_GALLERY_TABS: InvokeTabName[] = ['modelManager', 'queue'];
export const NO_SIDE_PANEL_TABS: InvokeTabName[] = ['modelManager', 'queue'];
const panelStyles: CSSProperties = { height: '100%', width: '100%' };
const GALLERY_MIN_SIZE_PCT = 20;
const OPTIONS_PANEL_WIDTH = '434px';

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
    ref: galleryPanelRef,
    minSize: galleryPanelMinSize,
    isCollapsed: isGalleryPanelCollapsed,
    onCollapse: onCollapseGalleryPanel,
    onExpand: onExpandGalleryPanel,
    reset: resetGalleryPanel,
    expand: expandGalleryPanel,
    toggle: toggleGalleryPanel,
  } = usePanel(GALLERY_MIN_SIZE_PCT);

  useHotkeys('g', toggleGalleryPanel, []);

  const panelStorage = usePanelStorage();

  return (
    <InvTabs
      variant="appTabs"
      defaultIndex={activeTabIndex}
      index={activeTabIndex}
      onChange={handleTabChange}
      flexGrow={1}
      gap={4}
      isLazy
    >
      <InvTabList gap={4} flexDir="column">
        {tabs}
        <Spacer />
      </InvTabList>
      <PanelGroup
        id="app"
        autoSaveId="app"
        direction="horizontal"
        style={panelStyles}
        storage={panelStorage}
      >
        <Panel id="main" order={0} minSize={50}>
          <Flex w="full" h="full" gap={4}>
            {!NO_SIDE_PANEL_TABS.includes(activeTabName) && (
              <Flex h="full" w={OPTIONS_PANEL_WIDTH} flexShrink={0}>
                {activeTabName === 'nodes' ? (
                  <NodeEditorPanelGroup />
                ) : (
                  <ParametersPanel />
                )}
              </Flex>
            )}
            <InvTabPanels w="full" h="full">
              {tabPanels}
            </InvTabPanels>
          </Flex>
        </Panel>
        {!NO_GALLERY_TABS.includes(activeTabName) && (
          <>
            <ResizeHandle onDoubleClick={resetGalleryPanel} />
            <Panel
              id="gallery"
              ref={galleryPanelRef}
              order={1}
              defaultSize={galleryPanelMinSize}
              minSize={galleryPanelMinSize}
              onCollapse={onCollapseGalleryPanel}
              onExpand={onExpandGalleryPanel}
              collapsible
            >
              <ImageGalleryContent />
            </Panel>
            <FloatingGalleryButton
              isGalleryCollapsed={isGalleryPanelCollapsed}
              expandGallery={expandGalleryPanel}
            />
          </>
        )}
      </PanelGroup>
    </InvTabs>
  );
};

export default memo(InvokeTabs);

import {
  ChakraProps,
  Icon,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  Tooltip,
  VisuallyHidden,
} from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { setIsLightboxOpen } from 'features/lightbox/store/lightboxSlice';
import { InvokeTabName } from 'features/ui/store/tabMap';
import { setActiveTab, togglePanels } from 'features/ui/store/uiSlice';
import { ReactNode, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { MdDeviceHub, MdGridOn } from 'react-icons/md';
import { activeTabIndexSelector } from '../store/uiSelectors';
import UnifiedCanvasWorkarea from 'features/ui/components/tabs/UnifiedCanvas/UnifiedCanvasWorkarea';
import { useTranslation } from 'react-i18next';
import { ResourceKey } from 'i18next';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import NodeEditor from 'features/nodes/components/NodeEditor';
import LinearWorkspace from './tabs/Linear/LinearWorkspace';
import { FaImage } from 'react-icons/fa';

export interface InvokeTabInfo {
  id: InvokeTabName;
  icon: ReactNode;
  workarea: ReactNode;
}

const tabIconStyles: ChakraProps['sx'] = {
  boxSize: 6,
};

const buildTabs = (disabledTabs: InvokeTabName[]): InvokeTabInfo[] => {
  const tabs: InvokeTabInfo[] = [
    {
      id: 'linear',
      icon: <Icon as={FaImage} sx={tabIconStyles} />,
      workarea: <LinearWorkspace />,
    },
    {
      id: 'unifiedCanvas',
      icon: <Icon as={MdGridOn} sx={tabIconStyles} />,
      workarea: <UnifiedCanvasWorkarea />,
    },
    {
      id: 'nodes',
      icon: <Icon as={MdDeviceHub} sx={tabIconStyles} />,
      workarea: <NodeEditor />,
    },
  ];
  return tabs.filter((tab) => !disabledTabs.includes(tab.id));
};

export default function InvokeTabs() {
  const activeTab = useAppSelector(activeTabIndexSelector);

  const isLightBoxOpen = useAppSelector(
    (state: RootState) => state.lightbox.isLightboxOpen
  );

  const { shouldPinGallery, shouldPinParametersPanel } = useAppSelector(
    (state: RootState) => state.ui
  );

  const disabledTabs = useAppSelector(
    (state: RootState) => state.system.disabledTabs
  );

  const activeTabs = buildTabs(disabledTabs);

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  useHotkeys('1', () => {
    dispatch(setActiveTab(0));
  });

  useHotkeys('2', () => {
    dispatch(setActiveTab(1));
  });

  useHotkeys('3', () => {
    dispatch(setActiveTab(2));
  });

  // Lightbox Hotkey
  useHotkeys(
    'z',
    () => {
      dispatch(setIsLightboxOpen(!isLightBoxOpen));
    },
    [isLightBoxOpen]
  );

  useHotkeys(
    'f',
    () => {
      dispatch(togglePanels());
      (shouldPinGallery || shouldPinParametersPanel) &&
        dispatch(requestCanvasRescale());
    },
    [shouldPinGallery, shouldPinParametersPanel]
  );

  const tabs = useMemo(
    () =>
      activeTabs.map((tab) => (
        <Tooltip
          key={tab.id}
          hasArrow
          label={String(t(`common.${tab.id}` as ResourceKey))}
          placement="end"
        >
          <Tab>
            <VisuallyHidden>
              {String(t(`common.${tab.id}` as ResourceKey))}
            </VisuallyHidden>
            {tab.icon}
          </Tab>
        </Tooltip>
      )),
    [t, activeTabs]
  );

  const tabPanels = useMemo(
    () =>
      activeTabs.map((tab) => <TabPanel key={tab.id}>{tab.workarea}</TabPanel>),
    [activeTabs]
  );

  return (
    <Tabs
      defaultIndex={activeTab}
      index={activeTab}
      onChange={(index: number) => {
        dispatch(setActiveTab(index));
      }}
      flexGrow={1}
      flexDir={{ base: 'column', xl: 'row' }}
      gap={{ base: 4 }}
      isLazy
    >
      <TabList
        pt={2}
        gap={4}
        flexDir={{ base: 'row', xl: 'column' }}
        justifyContent={{ base: 'center', xl: 'start' }}
      >
        {tabs}
      </TabList>
      <TabPanels>{tabPanels}</TabPanels>
    </Tabs>
  );
}

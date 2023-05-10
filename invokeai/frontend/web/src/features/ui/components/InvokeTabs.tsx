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
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setIsLightboxOpen } from 'features/lightbox/store/lightboxSlice';
import { InvokeTabName } from 'features/ui/store/tabMap';
import { setActiveTab, togglePanels } from 'features/ui/store/uiSlice';
import { memo, ReactNode, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { MdDeviceHub, MdGridOn } from 'react-icons/md';
import { activeTabIndexSelector } from '../store/uiSelectors';
import UnifiedCanvasWorkarea from 'features/ui/components/tabs/UnifiedCanvas/UnifiedCanvasWorkarea';
import { useTranslation } from 'react-i18next';
import { ResourceKey } from 'i18next';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import NodeEditor from 'features/nodes/components/NodeEditor';
import GenerateWorkspace from './tabs/Generate/GenerateWorkspace';
import { createSelector } from '@reduxjs/toolkit';
import { BsLightningChargeFill } from 'react-icons/bs';
import { configSelector } from 'features/system/store/configSelectors';
import { isEqual } from 'lodash';

export interface InvokeTabInfo {
  id: InvokeTabName;
  icon: ReactNode;
  workarea: ReactNode;
}

const tabs: InvokeTabInfo[] = [
  {
    id: 'generate',
    icon: <Icon as={BsLightningChargeFill} sx={{ boxSize: 5 }} />,
    workarea: <GenerateWorkspace />,
  },
  {
    id: 'unifiedCanvas',
    icon: <Icon as={MdGridOn} sx={{ boxSize: 6 }} />,
    workarea: <UnifiedCanvasWorkarea />,
  },
  {
    id: 'nodes',
    icon: <Icon as={MdDeviceHub} sx={{ boxSize: 6 }} />,
    workarea: <NodeEditor />,
  },
];

const enabledTabsSelector = createSelector(
  configSelector,
  (config) => {
    const { disabledTabs } = config;

    return tabs.filter((tab) => !disabledTabs.includes(tab.id));
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
  }
);

const InvokeTabs = () => {
  const activeTab = useAppSelector(activeTabIndexSelector);
  const enabledTabs = useAppSelector(enabledTabsSelector);
  const isLightBoxOpen = useAppSelector(
    (state: RootState) => state.lightbox.isLightboxOpen
  );

  const { shouldPinGallery, shouldPinParametersPanel } = useAppSelector(
    (state: RootState) => state.ui
  );

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  useHotkeys('1', () => {
    dispatch(setActiveTab('generate'));
  });

  useHotkeys('2', () => {
    dispatch(setActiveTab('unifiedCanvas'));
  });

  useHotkeys('3', () => {
    dispatch(setActiveTab('nodes'));
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
      enabledTabs.map((tab) => (
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
    [t, enabledTabs]
  );

  const tabPanels = useMemo(
    () =>
      enabledTabs.map((tab) => (
        <TabPanel key={tab.id}>{tab.workarea}</TabPanel>
      )),
    [enabledTabs]
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
};

export default memo(InvokeTabs);

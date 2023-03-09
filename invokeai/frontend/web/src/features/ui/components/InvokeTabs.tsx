import {
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
import NodesWIP from 'common/components/WorkInProgress/NodesWIP';
import { PostProcessingWIP } from 'common/components/WorkInProgress/PostProcessingWIP';
import TrainingWIP from 'common/components/WorkInProgress/Training';
import useUpdateTranslations from 'common/hooks/useUpdateTranslations';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import { setShouldShowGallery } from 'features/gallery/store/gallerySlice';
import Lightbox from 'features/lightbox/components/Lightbox';
import { setIsLightboxOpen } from 'features/lightbox/store/lightboxSlice';
import { InvokeTabName } from 'features/ui/store/tabMap';
import {
  setActiveTab,
  setShouldShowParametersPanel,
} from 'features/ui/store/uiSlice';
import i18n from 'i18n';
import { ReactElement } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import {
  MdDeviceHub,
  MdFlashOn,
  MdGridOn,
  MdPhotoFilter,
  MdPhotoLibrary,
  MdTextFields,
} from 'react-icons/md';
import { activeTabIndexSelector } from '../store/uiSelectors';
import { floatingSelector } from './FloatingParametersPanelButtons';
import ImageToImageWorkarea from './ImageToImage';
import TextToImageWorkarea from './TextToImage';
import UnifiedCanvasWorkarea from './UnifiedCanvas/UnifiedCanvasWorkarea';

export interface InvokeTabInfo {
  title: ReactElement;
  workarea: ReactElement;
  tooltip: string;
}

export const tabDict: Record<InvokeTabName, InvokeTabInfo> = {
  txt2img: {
    title: <Icon as={MdTextFields} boxSize={6} />,
    workarea: <TextToImageWorkarea />,
    tooltip: 'Text To Image',
  },
  img2img: {
    title: <Icon as={MdPhotoLibrary} boxSize={6} />,
    workarea: <ImageToImageWorkarea />,
    tooltip: 'Image To Image',
  },
  unifiedCanvas: {
    title: <Icon as={MdGridOn} boxSize={6} />,
    workarea: <UnifiedCanvasWorkarea />,
    tooltip: 'Unified Canvas',
  },
  nodes: {
    title: <Icon as={MdDeviceHub} boxSize={6} />,
    workarea: <NodesWIP />,
    tooltip: 'Nodes',
  },
  postprocess: {
    title: <Icon as={MdPhotoFilter} boxSize={6} />,
    workarea: <PostProcessingWIP />,
    tooltip: 'Post Processing',
  },
  training: {
    title: <Icon as={MdFlashOn} boxSize={6} />,
    workarea: <TrainingWIP />,
    tooltip: 'Training',
  },
};

function updateTabTranslations() {
  tabDict.txt2img.tooltip = i18n.t('common.text2img');
  tabDict.img2img.tooltip = i18n.t('common.img2img');
  tabDict.unifiedCanvas.tooltip = i18n.t('common.unifiedCanvas');
  tabDict.nodes.tooltip = i18n.t('common.nodes');
  tabDict.postprocess.tooltip = i18n.t('common.postProcessing');
  tabDict.training.tooltip = i18n.t('common.training');
}

export default function InvokeTabs() {
  const activeTab = useAppSelector(activeTabIndexSelector);

  const isLightBoxOpen = useAppSelector(
    (state: RootState) => state.lightbox.isLightboxOpen
  );

  const {
    shouldShowGallery,
    shouldShowParametersPanel,
    shouldPinGallery,
    shouldPinParametersPanel,
  } = useAppSelector(floatingSelector);

  useUpdateTranslations(updateTabTranslations);

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

  useHotkeys('4', () => {
    dispatch(setActiveTab(3));
  });

  useHotkeys('5', () => {
    dispatch(setActiveTab(4));
  });

  useHotkeys('6', () => {
    dispatch(setActiveTab(5));
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
      if (shouldShowGallery || shouldShowParametersPanel) {
        dispatch(setShouldShowParametersPanel(false));
        dispatch(setShouldShowGallery(false));
      } else {
        dispatch(setShouldShowParametersPanel(true));
        dispatch(setShouldShowGallery(true));
      }
      if (shouldPinGallery || shouldPinParametersPanel)
        setTimeout(() => dispatch(setDoesCanvasNeedScaling(true)), 400);
    },
    [shouldShowGallery, shouldShowParametersPanel]
  );

  const renderTabs = () => {
    const tabsToRender: ReactElement[] = [];
    Object.keys(tabDict).forEach((key) => {
      tabsToRender.push(
        <Tooltip
          key={key}
          hasArrow
          label={tabDict[key as keyof typeof tabDict].tooltip}
          placement="end"
        >
          <Tab>
            <VisuallyHidden>
              {tabDict[key as keyof typeof tabDict].tooltip}
            </VisuallyHidden>
            {tabDict[key as keyof typeof tabDict].title}
          </Tab>
        </Tooltip>
      );
    });
    return tabsToRender;
  };

  const renderTabPanels = () => {
    const tabPanelsToRender: ReactElement[] = [];
    Object.keys(tabDict).forEach((key) => {
      tabPanelsToRender.push(
        <TabPanel key={key}>
          {tabDict[key as keyof typeof tabDict].workarea}
        </TabPanel>
      );
    });
    return tabPanelsToRender;
  };

  return (
    <Tabs
      isLazy
      defaultIndex={activeTab}
      index={activeTab}
      onChange={(index: number) => {
        dispatch(setActiveTab(index));
      }}
    >
      <TabList>{renderTabs()}</TabList>
      <TabPanels>{isLightBoxOpen ? <Lightbox /> : renderTabPanels()}</TabPanels>
    </Tabs>
  );
}

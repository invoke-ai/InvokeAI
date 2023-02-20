import { Tab, TabPanel, TabPanels, Tabs, Tooltip } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import NodesWIP from 'common/components/WorkInProgress/NodesWIP';
import { PostProcessingWIP } from 'common/components/WorkInProgress/PostProcessingWIP';
import TrainingWIP from 'common/components/WorkInProgress/Training';
import useUpdateTranslations from 'common/hooks/useUpdateTranslations';
import ImageToImageIcon from 'common/icons/ImageToImageIcon';
import NodesIcon from 'common/icons/NodesIcon';
import PostprocessingIcon from 'common/icons/PostprocessingIcon';
import TextToImageIcon from 'common/icons/TextToImageIcon';
import TrainingIcon from 'common/icons/TrainingIcon';
import UnifiedCanvasIcon from 'common/icons/UnifiedCanvasIcon';
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
    title: <TextToImageIcon fill="black" boxSize="2.5rem" />,
    workarea: <TextToImageWorkarea />,
    tooltip: 'Text To Image',
  },
  img2img: {
    title: <ImageToImageIcon fill="black" boxSize="2.5rem" />,
    workarea: <ImageToImageWorkarea />,
    tooltip: 'Image To Image',
  },
  unifiedCanvas: {
    title: <UnifiedCanvasIcon fill="black" boxSize="2.5rem" />,
    workarea: <UnifiedCanvasWorkarea />,
    tooltip: 'Unified Canvas',
  },
  nodes: {
    title: <NodesIcon fill="black" boxSize="2.5rem" />,
    workarea: <NodesWIP />,
    tooltip: 'Nodes',
  },
  postprocess: {
    title: <PostprocessingIcon fill="black" boxSize="2.5rem" />,
    workarea: <PostProcessingWIP />,
    tooltip: 'Post Processing',
  },
  training: {
    title: <TrainingIcon fill="black" boxSize="2.5rem" />,
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
          placement="right"
        >
          <Tab>{tabDict[key as keyof typeof tabDict].title}</Tab>
        </Tooltip>
      );
    });
    return tabsToRender;
  };

  const renderTabPanels = () => {
    const tabPanelsToRender: ReactElement[] = [];
    Object.keys(tabDict).forEach((key) => {
      tabPanelsToRender.push(
        <TabPanel className="app-tabs-panel" key={key}>
          {tabDict[key as keyof typeof tabDict].workarea}
        </TabPanel>
      );
    });
    return tabPanelsToRender;
  };

  return (
    <Tabs
      isLazy
      className="app-tabs"
      variant="unstyled"
      defaultIndex={activeTab}
      index={activeTab}
      onChange={(index: number) => {
        dispatch(setActiveTab(index));
      }}
    >
      <div className="app-tabs-list">{renderTabs()}</div>
      <TabPanels className="app-tabs-panels">
        {isLightBoxOpen ? <Lightbox /> : renderTabPanels()}
      </TabPanels>
    </Tabs>
  );
}

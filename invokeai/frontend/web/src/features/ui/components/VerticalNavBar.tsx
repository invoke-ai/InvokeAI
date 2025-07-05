import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $customNavComponent } from 'app/store/nanostores/customNavComponent';
import { useAppSelector } from 'app/store/storeHooks';
import InvokeAILogoComponent from 'features/system/components/InvokeAILogoComponent';
import SettingsMenu from 'features/system/components/SettingsModal/SettingsMenu';
import StatusIndicator from 'features/system/components/StatusIndicator';
import { VideosModalButton } from 'features/system/components/VideosModal/VideosModalButton';
import {
  selectWithCanvasTab,
  selectWithGenerateTab,
  selectWithModelsTab,
  selectWithQueueTab,
  selectWithUpscalingTab,
  selectWithWorkflowsTab,
} from 'features/system/store/configSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiBoundingBoxBold,
  PiCubeBold,
  PiFlowArrowBold,
  PiFrameCornersBold,
  PiQueueBold,
  PiTextAaBold,
} from 'react-icons/pi';

import { Notifications } from './Notifications';
import { TabButton } from './TabButton';

export const VerticalNavBar = memo(() => {
  const { t } = useTranslation();
  const customNavComponent = useStore($customNavComponent);
  const withGenerateTab = useAppSelector(selectWithGenerateTab);
  const withCanvasTab = useAppSelector(selectWithCanvasTab);
  const withUpscalingTab = useAppSelector(selectWithUpscalingTab);
  const withWorkflowsTab = useAppSelector(selectWithWorkflowsTab);
  const withModelsTab = useAppSelector(selectWithModelsTab);
  const withQueueTab = useAppSelector(selectWithQueueTab);

  return (
    <Flex flexDir="column" alignItems="center" py={6} ps={4} pe={2} gap={4} minW={0} flexShrink={0}>
      <InvokeAILogoComponent />
      <Flex gap={6} pt={6} h="full" flexDir="column">
        {withGenerateTab && <TabButton tab="generate" icon={<PiTextAaBold />} label="Generate" />}
        {withCanvasTab && <TabButton tab="canvas" icon={<PiBoundingBoxBold />} label={t('ui.tabs.canvas')} />}
        {withUpscalingTab && <TabButton tab="upscaling" icon={<PiFrameCornersBold />} label={t('ui.tabs.upscaling')} />}
        {withWorkflowsTab && <TabButton tab="workflows" icon={<PiFlowArrowBold />} label={t('ui.tabs.workflows')} />}
        {withModelsTab && <TabButton tab="models" icon={<PiCubeBold />} label={t('ui.tabs.models')} />}
        {withQueueTab && <TabButton tab="queue" icon={<PiQueueBold />} label={t('ui.tabs.queue')} />}
      </Flex>
      <Spacer />
      <StatusIndicator />
      <Notifications />
      <VideosModalButton />
      {customNavComponent ? customNavComponent : <SettingsMenu />}
    </Flex>
  );
});

VerticalNavBar.displayName = 'VerticalNavBar';

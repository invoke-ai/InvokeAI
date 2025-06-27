import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $customNavComponent } from 'app/store/nanostores/customNavComponent';
import InvokeAILogoComponent from 'features/system/components/InvokeAILogoComponent';
import SettingsMenu from 'features/system/components/SettingsModal/SettingsMenu';
import StatusIndicator from 'features/system/components/StatusIndicator';
import { VideosModalButton } from 'features/system/components/VideosModal/VideosModalButton';
import { TabMountGate } from 'features/ui/components/TabMountGate';
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

  return (
    <Flex flexDir="column" alignItems="center" py={6} px={4} gap={4} minW={0} flexShrink={0}>
      <InvokeAILogoComponent />
      <Flex gap={6} pt={6} h="full" flexDir="column">
        <TabMountGate tab="generate">
          <TabButton tab="generate" icon={<PiTextAaBold />} label="Generate" />
        </TabMountGate>
        <TabMountGate tab="canvas">
          <TabButton tab="canvas" icon={<PiBoundingBoxBold />} label={t('ui.tabs.canvas')} />
        </TabMountGate>
        <TabMountGate tab="upscaling">
          <TabButton tab="upscaling" icon={<PiFrameCornersBold />} label={t('ui.tabs.upscaling')} />
        </TabMountGate>
        <TabMountGate tab="workflows">
          <TabButton tab="workflows" icon={<PiFlowArrowBold />} label={t('ui.tabs.workflows')} />
        </TabMountGate>
        <TabMountGate tab="models">
          <TabButton tab="models" icon={<PiCubeBold />} label={t('ui.tabs.models')} />
        </TabMountGate>
        <TabMountGate tab="queue">
          <TabButton tab="queue" icon={<PiQueueBold />} label={t('ui.tabs.queue')} />
        </TabMountGate>
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

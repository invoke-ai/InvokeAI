import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $customNavComponent } from 'app/store/nanostores/customNavComponent';
import InvokeAILogoComponent from 'features/system/components/InvokeAILogoComponent';
import SettingsMenu from 'features/system/components/SettingsModal/SettingsMenu';
import StatusIndicator from 'features/system/components/StatusIndicator';
import { TabMountGate } from 'features/ui/components/TabMountGate';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { MdZoomOutMap } from 'react-icons/md';
import { PiFlowArrowBold } from 'react-icons/pi';
import { RiBox2Line, RiInputMethodLine, RiPlayList2Fill } from 'react-icons/ri';

import { TabButton } from './TabButton';

export const VerticalNavBar = memo(() => {
  const { t } = useTranslation();
  const customNavComponent = useStore($customNavComponent);

  return (
    <Flex flexDir="column" alignItems="center" py={2} gap={4}>
      <InvokeAILogoComponent />
      <Flex gap={4} pt={6} h="full" flexDir="column">
        <TabMountGate tab="generation">
          <TabButton tab="generation" icon={<RiInputMethodLine />} label={t('ui.tabs.generation')} />
        </TabMountGate>
        <TabMountGate tab="upscaling">
          <TabButton tab="upscaling" icon={<MdZoomOutMap />} label={t('ui.tabs.upscaling')} />
        </TabMountGate>
        <TabMountGate tab="workflows">
          <TabButton tab="workflows" icon={<PiFlowArrowBold />} label={t('ui.tabs.workflows')} />
        </TabMountGate>
        <TabMountGate tab="models">
          <TabButton tab="models" icon={<RiBox2Line />} label={t('ui.tabs.models')} />
        </TabMountGate>
        <TabMountGate tab="queue">
          <TabButton tab="queue" icon={<RiPlayList2Fill />} label={t('ui.tabs.queue')} />
        </TabMountGate>
      </Flex>
      <Spacer />
      <StatusIndicator />
      {customNavComponent ? customNavComponent : <SettingsMenu />}
    </Flex>
  );
});

VerticalNavBar.displayName = 'VerticalNavBar';

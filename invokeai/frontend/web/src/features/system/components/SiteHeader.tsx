import { Flex, Spacer, useDisclosure } from '@chakra-ui/react';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { InvMenuList } from 'common/components/InvMenu/InvMenuList';
import {
  InvMenu,
  InvMenuButton,
  InvMenuGroup,
} from 'common/components/InvMenu/wrapper';
import { useGlobalMenuCloseTrigger } from 'common/hooks/useGlobalMenuCloseTrigger';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  FaBars,
  FaBug,
  FaCog,
  FaDiscord,
  FaGithub,
  FaKeyboard,
} from 'react-icons/fa';

import HotkeysModal from './HotkeysModal/HotkeysModal';
import InvokeAILogoComponent from './InvokeAILogoComponent';
import SettingsModal from './SettingsModal/SettingsModal';
import StatusIndicator from './StatusIndicator';

const SiteHeader = () => {
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  useGlobalMenuCloseTrigger(onClose);

  const isBugLinkEnabled = useFeatureStatus('bugLink').isFeatureEnabled;
  const isDiscordLinkEnabled = useFeatureStatus('discordLink').isFeatureEnabled;
  const isGithubLinkEnabled = useFeatureStatus('githubLink').isFeatureEnabled;

  const githubLink = 'http://github.com/invoke-ai/InvokeAI';
  const discordLink = 'https://discord.gg/ZmtBAhwWhy';

  return (
    <Flex gap={2} alignItems="center">
      <InvokeAILogoComponent />
      <Spacer />
      <StatusIndicator />

      <InvMenu isOpen={isOpen} onOpen={onOpen} onClose={onClose}>
        <InvMenuButton
          as={InvIconButton}
          variant="link"
          aria-label={t('accessibility.menu')}
          icon={<FaBars />}
          boxSize={8}
        />
        <InvMenuList>
          <InvMenuGroup title={t('common.communityLabel')}>
            {isGithubLinkEnabled && (
              <InvMenuItem
                as="a"
                href={githubLink}
                target="_blank"
                icon={<FaGithub />}
              >
                {t('common.githubLabel')}
              </InvMenuItem>
            )}
            {isBugLinkEnabled && (
              <InvMenuItem
                as="a"
                href={`${githubLink}/issues`}
                target="_blank"
                icon={<FaBug />}
              >
                {t('common.reportBugLabel')}
              </InvMenuItem>
            )}
            {isDiscordLinkEnabled && (
              <InvMenuItem
                as="a"
                href={discordLink}
                target="_blank"
                icon={<FaDiscord />}
              >
                {t('common.discordLabel')}
              </InvMenuItem>
            )}
          </InvMenuGroup>
          <InvMenuGroup title={t('common.settingsLabel')}>
            <HotkeysModal>
              <InvMenuItem as="button" icon={<FaKeyboard />}>
                {t('common.hotkeysLabel')}
              </InvMenuItem>
            </HotkeysModal>
            <SettingsModal>
              <InvMenuItem as="button" icon={<FaCog />}>
                {t('common.settingsLabel')}
              </InvMenuItem>
            </SettingsModal>
          </InvMenuGroup>
        </InvMenuList>
      </InvMenu>
    </Flex>
  );
};

export default memo(SiteHeader);

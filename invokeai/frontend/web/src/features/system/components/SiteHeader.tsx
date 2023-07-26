import { Flex, Link, MenuGroup, Spacer } from '@chakra-ui/react';
import { memo } from 'react';
import StatusIndicator from './StatusIndicator';

import { Menu, MenuButton, MenuItem, MenuList } from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import { useTranslation } from 'react-i18next';
import { FaBars, FaBug, FaDiscord, FaKeyboard } from 'react-icons/fa';
import { MdSettings } from 'react-icons/md';
import { useFeatureStatus } from '../hooks/useFeatureStatus';
import HotkeysModal from './HotkeysModal/HotkeysModal';
import InvokeAILogoComponent from './InvokeAILogoComponent';
import SettingsModal from './SettingsModal/SettingsModal';

const SiteHeader = () => {
  const { t } = useTranslation();

  const isBugLinkEnabled = useFeatureStatus('bugLink').isFeatureEnabled;
  const isDiscordLinkEnabled = useFeatureStatus('discordLink').isFeatureEnabled;
  const isGithubLinkEnabled = useFeatureStatus('githubLink').isFeatureEnabled;

  const githubLink = 'http://github.com/invoke-ai/InvokeAI';
  const discordLink = 'https://discord.gg/ZmtBAhwWhy';

  return (
    <Flex
      sx={{
        gap: 2,
        alignItems: 'center',
      }}
    >
      <InvokeAILogoComponent />
      <Spacer />
      <StatusIndicator />

      <Menu>
        <MenuButton
          as={IAIIconButton}
          variant="link"
          aria-label={t('accessibility.menu')}
          icon={<FaBars />}
        />
        <MenuList>
          <MenuGroup title={t('common.communityLabel')}>
            {isGithubLinkEnabled && (
              <Link href={githubLink} target="_blank">
                <MenuItem icon={<FaDiscord />}>
                  {t('common.githubLabel')}
                </MenuItem>
              </Link>
            )}
            {isBugLinkEnabled && (
              <Link href={`${githubLink}/issues`} target="_blank">
                <MenuItem icon={<FaBug />}>
                  {t('common.reportBugLabel')}
                </MenuItem>
              </Link>
            )}
            {isDiscordLinkEnabled && (
              <Link href={discordLink} target="_blank">
                <MenuItem icon={<FaDiscord />}>
                  {t('common.discordLabel')}
                </MenuItem>
              </Link>
            )}
          </MenuGroup>
          <MenuGroup title={t('common.settingsLabel')}>
            <HotkeysModal>
              <MenuItem as="button" icon={<FaKeyboard />}>
                {t('common.hotkeysLabel')}
              </MenuItem>
            </HotkeysModal>
            <SettingsModal>
              <MenuItem as="button" icon={<MdSettings />}>
                {t('common.settingsLabel')}
              </MenuItem>
            </SettingsModal>
          </MenuGroup>
        </MenuList>
      </Menu>
    </Flex>
  );
};

export default memo(SiteHeader);

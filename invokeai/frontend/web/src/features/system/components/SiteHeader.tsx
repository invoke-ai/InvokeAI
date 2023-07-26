import {
  Flex,
  Menu,
  MenuButton,
  MenuGroup,
  MenuItem,
  MenuList,
  Spacer,
} from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
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
import { menuListMotionProps } from 'theme/components/menu';
import { useFeatureStatus } from '../hooks/useFeatureStatus';
import HotkeysModal from './HotkeysModal/HotkeysModal';
import InvokeAILogoComponent from './InvokeAILogoComponent';
import SettingsModal from './SettingsModal/SettingsModal';
import StatusIndicator from './StatusIndicator';

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
          sx={{ boxSize: 8 }}
        />
        <MenuList motionProps={menuListMotionProps}>
          <MenuGroup title={t('common.communityLabel')}>
            {isGithubLinkEnabled && (
              <MenuItem
                as="a"
                href={githubLink}
                target="_blank"
                icon={<FaGithub />}
              >
                {t('common.githubLabel')}
              </MenuItem>
            )}
            {isBugLinkEnabled && (
              <MenuItem
                as="a"
                href={`${githubLink}/issues`}
                target="_blank"
                icon={<FaBug />}
              >
                {t('common.reportBugLabel')}
              </MenuItem>
            )}
            {isDiscordLinkEnabled && (
              <MenuItem
                as="a"
                href={discordLink}
                target="_blank"
                icon={<FaDiscord />}
              >
                {t('common.discordLabel')}
              </MenuItem>
            )}
          </MenuGroup>
          <MenuGroup title={t('common.settingsLabel')}>
            <HotkeysModal>
              <MenuItem as="button" icon={<FaKeyboard />}>
                {t('common.hotkeysLabel')}
              </MenuItem>
            </HotkeysModal>
            <SettingsModal>
              <MenuItem as="button" icon={<FaCog />}>
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

import {
  Menu,
  MenuButton,
  MenuGroup,
  MenuItem,
  MenuList,
  useDisclosure,
} from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import { useGlobalMenuCloseTrigger } from 'common/hooks/useGlobalMenuCloseTrigger';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
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
import HotkeysModal from './HotkeysModal/HotkeysModal';
import SettingsModal from './SettingsModal/SettingsModal';

export default function SettingsMenu() {
  const { t } = useTranslation();

  const { isOpen, onOpen, onClose } = useDisclosure();
  useGlobalMenuCloseTrigger(onClose);

  const isBugLinkEnabled = useFeatureStatus('bugLink').isFeatureEnabled;
  const isDiscordLinkEnabled = useFeatureStatus('discordLink').isFeatureEnabled;
  const isGithubLinkEnabled = useFeatureStatus('githubLink').isFeatureEnabled;

  const githubLink = 'http://github.com/invoke-ai/InvokeAI';
  const discordLink = 'https://discord.gg/ZmtBAhwWhy';

  return (
    <Menu isOpen={isOpen} onOpen={onOpen} onClose={onClose}>
      <MenuButton
        as={IAIIconButton}
        variant="link"
        aria-label={t('accessibility.menu')}
        icon={<FaBars size={16} />}
        sx={{ boxSize: 10 }}
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
  );
}

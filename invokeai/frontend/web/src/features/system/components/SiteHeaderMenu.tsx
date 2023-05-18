import { Flex, Link } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import { FaCube, FaKeyboard, FaBug, FaGithub, FaDiscord } from 'react-icons/fa';
import { MdSettings } from 'react-icons/md';
import HotkeysModal from './HotkeysModal/HotkeysModal';
import LanguagePicker from './LanguagePicker';
import ModelManagerModal from './ModelManager/ModelManagerModal';
import SettingsModal from './SettingsModal/SettingsModal';
import ThemeChanger from './ThemeChanger';
import IAIIconButton from 'common/components/IAIIconButton';
import { useFeatureStatus } from '../hooks/useFeatureStatus';

const SiteHeaderMenu = () => {
  const { t } = useTranslation();

  const isModelManagerEnabled =
    useFeatureStatus('modelManager').isFeatureEnabled;
  const isLocalizationEnabled =
    useFeatureStatus('localization').isFeatureEnabled;
  const isBugLinkEnabled = useFeatureStatus('bugLink').isFeatureEnabled;
  const isDiscordLinkEnabled = useFeatureStatus('discordLink').isFeatureEnabled;
  const isGithubLinkEnabled = useFeatureStatus('githubLink').isFeatureEnabled;

  return (
    <Flex
      alignItems="center"
      flexDirection={{ base: 'column', xl: 'row' }}
      gap={{ base: 4, xl: 1 }}
    >
      {isModelManagerEnabled && (
        <ModelManagerModal>
          <IAIIconButton
            aria-label={t('modelManager.modelManager')}
            tooltip={t('modelManager.modelManager')}
            size="sm"
            variant="link"
            data-variant="link"
            fontSize={20}
            icon={<FaCube />}
          />
        </ModelManagerModal>
      )}

      <HotkeysModal>
        <IAIIconButton
          aria-label={t('common.hotkeysLabel')}
          tooltip={t('common.hotkeysLabel')}
          size="sm"
          variant="link"
          data-variant="link"
          fontSize={20}
          icon={<FaKeyboard />}
        />
      </HotkeysModal>

      <ThemeChanger />

      {isLocalizationEnabled && <LanguagePicker />}

      {isBugLinkEnabled && (
        <Link
          isExternal
          href="http://github.com/invoke-ai/InvokeAI/issues"
          marginBottom="-0.25rem"
        >
          <IAIIconButton
            aria-label={t('common.reportBugLabel')}
            tooltip={t('common.reportBugLabel')}
            variant="link"
            data-variant="link"
            fontSize={20}
            size="sm"
            icon={<FaBug />}
          />
        </Link>
      )}

      {isGithubLinkEnabled && (
        <Link
          isExternal
          href="http://github.com/invoke-ai/InvokeAI"
          marginBottom="-0.25rem"
        >
          <IAIIconButton
            aria-label={t('common.githubLabel')}
            tooltip={t('common.githubLabel')}
            variant="link"
            data-variant="link"
            fontSize={20}
            size="sm"
            icon={<FaGithub />}
          />
        </Link>
      )}

      {isDiscordLinkEnabled && (
        <Link
          isExternal
          href="https://discord.gg/ZmtBAhwWhy"
          marginBottom="-0.25rem"
        >
          <IAIIconButton
            aria-label={t('common.discordLabel')}
            tooltip={t('common.discordLabel')}
            variant="link"
            data-variant="link"
            fontSize={20}
            size="sm"
            icon={<FaDiscord />}
          />
        </Link>
      )}

      <SettingsModal>
        <IAIIconButton
          aria-label={t('common.settingsLabel')}
          tooltip={t('common.settingsLabel')}
          variant="link"
          data-variant="link"
          fontSize={22}
          size="sm"
          icon={<MdSettings />}
        />
      </SettingsModal>
    </Flex>
  );
};

SiteHeaderMenu.displayName = 'SiteHeaderMenu';
export default SiteHeaderMenu;

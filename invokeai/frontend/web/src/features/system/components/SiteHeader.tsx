import { Flex, Spacer } from '@chakra-ui/react';
import { memo } from 'react';
import StatusIndicator from './StatusIndicator';

import { Link } from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import { useTranslation } from 'react-i18next';
import { FaBug, FaCube, FaDiscord, FaGithub, FaKeyboard } from 'react-icons/fa';
import { MdSettings } from 'react-icons/md';
import HotkeysModal from './HotkeysModal/HotkeysModal';
import InvokeAILogoComponent from './InvokeAILogoComponent';
import LanguagePicker from './LanguagePicker';
import ModelManagerModal from './ModelManager/ModelManagerModal';
import SettingsModal from './SettingsModal/SettingsModal';
import ThemeChanger from './ThemeChanger';
import { useFeatureStatus } from '../hooks/useFeatureStatus';

const SiteHeader = () => {
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
      sx={{
        gap: 2,
        alignItems: 'center',
      }}
    >
      <InvokeAILogoComponent />
      <Spacer />
      <StatusIndicator />

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

export default memo(SiteHeader);

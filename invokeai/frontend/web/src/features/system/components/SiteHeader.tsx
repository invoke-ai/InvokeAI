import { Flex, Grid, Link } from '@chakra-ui/react';

import { FaBug, FaCube, FaDiscord, FaGithub, FaKeyboard } from 'react-icons/fa';

import IAIIconButton from 'common/components/IAIIconButton';

import HotkeysModal from './HotkeysModal/HotkeysModal';

import ModelManagerModal from './ModelManager/ModelManagerModal';
import ModelSelect from './ModelSelect';
import SettingsModal from './SettingsModal/SettingsModal';
import StatusIndicator from './StatusIndicator';
import ThemeChanger from './ThemeChanger';

import LanguagePicker from './LanguagePicker';

import { useTranslation } from 'react-i18next';
import { MdSettings } from 'react-icons/md';
import InvokeAILogoComponent from './InvokeAILogoComponent';

/**
 * Header, includes color mode toggle, settings button, status message.
 */
const SiteHeader = () => {
  const { t } = useTranslation();

  return (
    <Grid gridTemplateColumns="auto max-content">
      <InvokeAILogoComponent />

      <Flex alignItems="center" gap={2}>
        <StatusIndicator />

        <ModelSelect />

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

        <LanguagePicker />

        <IAIIconButton
          aria-label={t('common.reportBugLabel')}
          tooltip={t('common.reportBugLabel')}
          variant="link"
          data-variant="link"
          fontSize={20}
          size="sm"
          icon={
            <Link isExternal href="http://github.com/invoke-ai/InvokeAI/issues">
              <FaBug />
            </Link>
          }
        />

        <IAIIconButton
          aria-label={t('common.githubLabel')}
          tooltip={t('common.githubLabel')}
          variant="link"
          data-variant="link"
          fontSize={20}
          size="sm"
          icon={
            <Link isExternal href="http://github.com/invoke-ai/InvokeAI">
              <FaGithub />
            </Link>
          }
        />

        <IAIIconButton
          aria-label={t('common.discordLabel')}
          tooltip={t('common.discordLabel')}
          variant="link"
          data-variant="link"
          fontSize={20}
          size="sm"
          icon={
            <Link isExternal href="https://discord.gg/ZmtBAhwWhy">
              <FaDiscord />
            </Link>
          }
        />

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
    </Grid>
  );
};

export default SiteHeader;

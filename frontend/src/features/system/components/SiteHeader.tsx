import { Link } from '@chakra-ui/react';

import {
  FaGithub,
  FaDiscord,
  FaBug,
  FaKeyboard,
  FaWrench,
} from 'react-icons/fa';

import InvokeAILogo from 'assets/images/logo.png';
import IAIIconButton from 'common/components/IAIIconButton';

import HotkeysModal from './HotkeysModal/HotkeysModal';

import SettingsModal from './SettingsModal/SettingsModal';
import StatusIndicator from './StatusIndicator';
import ThemeChanger from './ThemeChanger';
import ModelSelect from './ModelSelect';

import { useTranslation } from 'react-i18next';
import IAISelect from 'common/components/IAISelect';

/**
 * Header, includes color mode toggle, settings button, status message.
 */
const SiteHeader = () => {
  const { t, i18n } = useTranslation();

  return (
    <div className="site-header">
      <div className="site-header-left-side">
        <img src={InvokeAILogo} alt="invoke-ai-logo" />
        <h1>
          invoke <strong>ai</strong>
        </h1>
      </div>

      <div className="site-header-right-side">
        <StatusIndicator />

        <ModelSelect />

        <IAISelect
          validValues={['en', 'ru']}
          value={localStorage.getItem('i18nextLng') || 'en'}
          onChange={(e) => i18n.changeLanguage(e.target.value)}
        />

        <HotkeysModal>
          <IAIIconButton
            aria-label={t('common:hotkeysLabel')}
            tooltip={t('common:hotkeysLabel')}
            size={'sm'}
            variant="link"
            data-variant="link"
            fontSize={20}
            icon={<FaKeyboard />}
          />
        </HotkeysModal>

        <ThemeChanger />

        <IAIIconButton
          aria-label={t('common:reportBugLabel')}
          tooltip={t('common:reportBugLabel')}
          variant="link"
          data-variant="link"
          fontSize={20}
          size={'sm'}
          icon={
            <Link isExternal href="http://github.com/invoke-ai/InvokeAI/issues">
              <FaBug />
            </Link>
          }
        />

        <IAIIconButton
          aria-label={t('common:githubLabel')}
          tooltip={t('common:githubLabel')}
          variant="link"
          data-variant="link"
          fontSize={20}
          size={'sm'}
          icon={
            <Link isExternal href="http://github.com/invoke-ai/InvokeAI">
              <FaGithub />
            </Link>
          }
        />

        <IAIIconButton
          aria-label={t('common:discordLabel')}
          tooltip={t('common:discordLabel')}
          variant="link"
          data-variant="link"
          fontSize={20}
          size={'sm'}
          icon={
            <Link isExternal href="https://discord.gg/ZmtBAhwWhy">
              <FaDiscord />
            </Link>
          }
        />

        <SettingsModal>
          <IAIIconButton
            aria-label={t('common:settingsLabel')}
            tooltip={t('common:settingsLabel')}
            variant="link"
            data-variant="link"
            fontSize={20}
            size={'sm'}
            icon={<FaWrench />}
          />
        </SettingsModal>
      </div>
    </div>
  );
};

export default SiteHeader;

import { Link } from '@chakra-ui/react';

import { FaGithub, FaDiscord, FaBug, FaKeyboard, FaCube } from 'react-icons/fa';

import InvokeAILogo from 'assets/images/logo.png';
import IAIIconButton from 'common/components/IAIIconButton';

import HotkeysModal from './HotkeysModal/HotkeysModal';

import SettingsModal from './SettingsModal/SettingsModal';
import StatusIndicator from './StatusIndicator';
import ThemeChanger from './ThemeChanger';
import ModelSelect from './ModelSelect';
import ModelManagerModal from './ModelManager/ModelManagerModal';

import LanguagePicker from './LanguagePicker';

import { useTranslation } from 'react-i18next';
import { MdSettings } from 'react-icons/md';

/**
 * Header, includes color mode toggle, settings button, status message.
 */
const SiteHeader = () => {
  const { t } = useTranslation();

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

        <ModelManagerModal>
          <IAIIconButton
            aria-label={t('modelmanager:modelManager')}
            tooltip={t('modelmanager:modelManager')}
            size={'sm'}
            variant="link"
            data-variant="link"
            fontSize={20}
            icon={<FaCube />}
          />
        </ModelManagerModal>

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

        <LanguagePicker />

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
            fontSize={22}
            size={'sm'}
            icon={<MdSettings />}
          />
        </SettingsModal>
      </div>
    </div>
  );
};

export default SiteHeader;

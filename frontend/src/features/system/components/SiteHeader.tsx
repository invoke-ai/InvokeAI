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

/**
 * Header, includes color mode toggle, settings button, status message.
 */
const SiteHeader = () => {
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

        <HotkeysModal>
          <IAIIconButton
            aria-label="Hotkeys"
            tooltip="Hotkeys"
            size={'sm'}
            variant="link"
            data-variant="link"
            fontSize={20}
            icon={<FaKeyboard />}
          />
        </HotkeysModal>

        <ThemeChanger />

        <IAIIconButton
          aria-label="Report Bug"
          tooltip="Report Bug"
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
          aria-label="Link to Github Repo"
          tooltip="Github"
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
          aria-label="Link to Discord Server"
          tooltip="Discord"
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
            aria-label="Settings"
            tooltip="Settings"
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

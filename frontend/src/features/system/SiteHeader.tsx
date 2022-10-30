import { IconButton, Link, Tooltip, useColorMode } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { useHotkeys } from 'react-hotkeys-hook';

import { FaSun, FaMoon, FaGithub, FaDiscord } from 'react-icons/fa';
import { MdHelp, MdKeyboard, MdSettings } from 'react-icons/md';
import { RootState, useAppSelector } from '../../app/store';

import InvokeAILogo from '../../assets/images/logo.png';
import { OptionsState } from '../options/optionsSlice';
import CancelButton from '../options/ProcessButtons/CancelButton';
import InvokeButton from '../options/ProcessButtons/InvokeButton';
import ProcessButtons from '../options/ProcessButtons/ProcessButtons';
import HotkeysModal from './HotkeysModal/HotkeysModal';

import SettingsModal from './SettingsModal/SettingsModal';
import StatusIndicator from './StatusIndicator';

const siteHeaderSelector = createSelector(
  (state: RootState) => state.options,

  (options: OptionsState) => {
    const { shouldPinOptionsPanel } = options;
    return { shouldShowProcessButtons: !shouldPinOptionsPanel };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

/**
 * Header, includes color mode toggle, settings button, status message.
 */
const SiteHeader = () => {
  const { shouldShowProcessButtons } = useAppSelector(siteHeaderSelector);
  const { colorMode, toggleColorMode } = useColorMode();

  useHotkeys(
    'shift+d',
    () => {
      toggleColorMode();
    },
    [colorMode, toggleColorMode]
  );

  const colorModeIcon = colorMode == 'light' ? <FaMoon /> : <FaSun />;

  // Make FaMoon and FaSun icon apparent size consistent
  const colorModeIconFontSize = colorMode == 'light' ? 18 : 20;

  return (
    <div className="site-header">
      <div className="site-header-left-side">
        <img src={InvokeAILogo} alt="invoke-ai-logo" />
        <h1>
          invoke <strong>ai</strong>
        </h1>
        {shouldShowProcessButtons && (
          <div className="process-buttons process-buttons">
            <InvokeButton size={'sm'} />
            <CancelButton size={'sm'} />
          </div>
        )}
      </div>

      <div className="site-header-right-side">
        <StatusIndicator />

        <HotkeysModal>
          <IconButton
            aria-label="Hotkeys"
            variant="link"
            fontSize={24}
            size={'sm'}
            icon={<MdKeyboard />}
          />
        </HotkeysModal>

        <Tooltip hasArrow label="Theme" placement={'bottom'}>
          <IconButton
            aria-label="Toggle Dark Mode"
            onClick={toggleColorMode}
            variant="link"
            size={'sm'}
            fontSize={colorModeIconFontSize}
            icon={colorModeIcon}
          />
        </Tooltip>

        <Tooltip hasArrow label="Report Bug" placement={'bottom'}>
          <IconButton
            aria-label="Link to Github Issues"
            variant="link"
            fontSize={23}
            size={'sm'}
            icon={
              <Link
                isExternal
                href="http://github.com/invoke-ai/InvokeAI/issues"
              >
                <MdHelp />
              </Link>
            }
          />
        </Tooltip>

        <Tooltip hasArrow label="Github" placement={'bottom'}>
          <IconButton
            aria-label="Link to Github Repo"
            variant="link"
            fontSize={20}
            size={'sm'}
            icon={
              <Link isExternal href="http://github.com/invoke-ai/InvokeAI">
                <FaGithub />
              </Link>
            }
          />
        </Tooltip>

        <Tooltip hasArrow label="Discord" placement={'bottom'}>
          <IconButton
            aria-label="Link to Discord Server"
            variant="link"
            fontSize={20}
            size={'sm'}
            icon={
              <Link isExternal href="https://discord.gg/ZmtBAhwWhy">
                <FaDiscord />
              </Link>
            }
          />
        </Tooltip>

        <SettingsModal>
          <IconButton
            aria-label="Settings"
            variant="link"
            fontSize={24}
            size={'sm'}
            icon={<MdSettings />}
          />
        </SettingsModal>
      </div>
    </div>
  );
};

export default SiteHeader;

import {
  Flex,
  Heading,
  IconButton,
  Link,
  Spacer,
  Text,
  useColorMode,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';

import { FaSun, FaMoon, FaGithub } from 'react-icons/fa';
import { MdHelp, MdSettings } from 'react-icons/md';
import { useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import SettingsModal from '../system/SettingsModal';
import { SystemState } from '../system/systemSlice';
import InvokeAILogo from '../../assets/images/logo.png';

const systemSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => {
    return {
      isConnected: system.isConnected,
      isProcessing: system.isProcessing,
      currentIteration: system.currentIteration,
      totalIterations: system.totalIterations,
      currentStatus: system.currentStatus,
    };
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
  }
);

/**
 * Header, includes color mode toggle, settings button, status message.
 */
const SiteHeader = () => {
  const { colorMode, toggleColorMode } = useColorMode();
  const {
    isConnected,
    isProcessing,
    currentIteration,
    totalIterations,
    currentStatus,
  } = useAppSelector(systemSelector);

  const statusMessageTextColor = isConnected ? 'green.500' : 'red.500';

  const colorModeIcon = colorMode == 'light' ? <FaMoon /> : <FaSun />;

  // Make FaMoon and FaSun icon apparent size consistent
  const colorModeIconFontSize = colorMode == 'light' ? 18 : 20;

  let statusMessage = currentStatus;

  if (isProcessing) {
    if (totalIterations > 1) {
      statusMessage += ` [${currentIteration}/${totalIterations}]`;
    }
  }

  return (
    <div className="site-header">
      <div className="site-header-left-side">
        <img src={InvokeAILogo} alt="invoke-ai-logo" />
        <h1>
          invoke <strong>ai</strong>
        </h1>
      </div>

      <div className="site-header-right-side">
        <Text textColor={statusMessageTextColor}>{statusMessage}</Text>

        <SettingsModal>
          <IconButton
            aria-label="Settings"
            variant="link"
            fontSize={24}
            size={'sm'}
            icon={<MdSettings />}
          />
        </SettingsModal>

        <IconButton
          aria-label="Link to Github Issues"
          variant="link"
          fontSize={23}
          size={'sm'}
          icon={
            <Link
              isExternal
              href="http://github.com/lstein/stable-diffusion/issues"
            >
              <MdHelp />
            </Link>
          }
        />

        <IconButton
          aria-label="Link to Github Repo"
          variant="link"
          fontSize={20}
          size={'sm'}
          icon={
            <Link isExternal href="http://github.com/lstein/stable-diffusion">
              <FaGithub />
            </Link>
          }
        />

        <IconButton
          aria-label="Toggle Dark Mode"
          onClick={toggleColorMode}
          variant="link"
          size={'sm'}
          fontSize={colorModeIconFontSize}
          icon={colorModeIcon}
        />
      </div>
    </div>
  );
};

export default SiteHeader;

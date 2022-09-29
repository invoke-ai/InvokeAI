import {
  Flex,
  Heading,
  IconButton,
  Link,
  Spacer,
  useColorMode,
} from '@chakra-ui/react';

import { FaSun, FaMoon, FaGithub } from 'react-icons/fa';
import { MdHelp, MdSettings } from 'react-icons/md';
import SettingsModal from '../system/SettingsModal';
import StatusIndicator from './StatusIndicator';

/**
 * Header, includes color mode toggle, settings button, status message.
 */
const SiteHeader = () => {
  const { colorMode, toggleColorMode } = useColorMode();

  const colorModeIcon = colorMode == 'light' ? <FaMoon /> : <FaSun />;

  // Make FaMoon and FaSun icon apparent size consistent
  const colorModeIconFontSize = colorMode == 'light' ? 18 : 20;

  return (
    <Flex minWidth="max-content" alignItems="center" gap="1" pl={2} pr={1}>
      <Heading size={'lg'}>InvokeUI</Heading>

      <Spacer />

      <StatusIndicator />

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
    </Flex>
  );
};

export default SiteHeader;

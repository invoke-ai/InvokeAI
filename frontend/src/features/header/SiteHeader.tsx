import {
  Flex,
  Heading,
  IconButton,
  Link,
  Spacer,
  Text,
  useColorMode,
} from '@chakra-ui/react';

import { FaSun, FaMoon, FaGithub } from 'react-icons/fa';
import { MdHelp } from 'react-icons/md';
import { useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import SettingsModalButton from '../system/SettingsModalButton';

const SiteHeader = () => {
  const { colorMode, toggleColorMode } = useColorMode();
  const { isConnected } = useAppSelector((state: RootState) => state.system);

  return (
    <Flex minWidth='max-content' alignItems='center' gap='1'>
      <Heading size={'lg'}>Stable Diffusion Dream Server</Heading>

      <Spacer />

      <Text textColor={isConnected ? 'green.500' : 'red.500'}>
        {isConnected ? `Connected to server` : 'No connection to server'}
      </Text>

      <SettingsModalButton />

      <IconButton
        aria-label='Link to Github Issues'
        variant='link'
        fontSize={23}
        size={'sm'}
        icon={
          <Link
            isExternal
            href='http://github.com/lstein/stable-diffusion/issues'
          >
            <MdHelp />
          </Link>
        }
      />

      <IconButton
        aria-label='Link to Github Repo'
        variant='link'
        fontSize={20}
        size={'sm'}
        icon={
          <Link isExternal href='http://github.com/lstein/stable-diffusion'>
            <FaGithub />
          </Link>
        }
      />

      <IconButton
        aria-label='Toggle Dark Mode'
        onClick={toggleColorMode}
        variant='link'
        size={'sm'}
        fontSize={colorMode == 'light' ? 18 : 20}
        icon={colorMode == 'light' ? <FaMoon /> : <FaSun />}
      />
    </Flex>
  );
};

export default SiteHeader;

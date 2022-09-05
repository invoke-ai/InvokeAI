import {
  Flex,
  Heading,
  Icon,
  IconButton,
  Link,
  Spacer,
  useColorMode,
} from '@chakra-ui/react';

import { FaSun, FaMoon, FaGithub, FaCircle } from 'react-icons/fa';
import { MdHelp } from 'react-icons/md';
import { useAppSelector } from '../app/hooks';
import { RootState } from '../app/store';
import SettingsModalButton from '../features/settings/SettingsModalButton';

const SiteHeader = () => {
  const { colorMode, toggleColorMode } = useColorMode();
  const { isConnected } = useAppSelector((state: RootState) => state.sd);

  return (
    <Flex minWidth='max-content' alignItems='center' gap='1'>
      <Heading size={'lg'}>Stable Diffusion Dream Server</Heading>

      <Spacer />

      <IconButton
        size={'sm'}
        variant='link'
        fontSize={20}
        mt='1px'
        aria-label='Connection Status'
        icon={<FaCircle />}
        color={isConnected ? 'green.500' : 'red.500'}
      />

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

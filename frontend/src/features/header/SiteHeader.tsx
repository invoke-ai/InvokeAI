import {
  Flex,
  Heading,
  IconButton,
  Link,
  Spacer,
  Text,
  useColorMode,
} from '@chakra-ui/react';
import { useState } from 'react';

import { FaSun, FaMoon, FaGithub, FaServer } from 'react-icons/fa';
import { MdHelp } from 'react-icons/md';
import { useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import SettingsModalButton from '../system/SettingsModalButton';

const SiteHeader = () => {
  const { colorMode, toggleColorMode } = useColorMode();
  const { isConnected, socketId } = useAppSelector(
    (state: RootState) => state.system
  );
  const [isStatusIconHovered, setIsStatusIconHovered] =
    useState<boolean>(false);

  return (
    <Flex minWidth='max-content' alignItems='center' gap='1'>
      <Heading size={'lg'}>Stable Diffusion Dream Server</Heading>

      <Spacer />

      {isStatusIconHovered && (
        <Text textColor={isConnected ? 'green.500' : 'red.500'}>
          {isConnected ? `Session ID: ${socketId}` : 'No Connection'}
        </Text>
      )}
      <IconButton
        size={'sm'}
        variant='link'
        fontSize={20}
        mt='1px'
        aria-label='Connection Status'
        icon={<FaServer />}
        color={isConnected ? 'green.500' : 'red.500'}
        onMouseOver={() => setIsStatusIconHovered(true)}
        onMouseOut={() => setIsStatusIconHovered(false)}
        cursor='unset'
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

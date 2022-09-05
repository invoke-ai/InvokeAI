import {
  Flex,
  Heading,
  Icon,
  IconButton,
  Link,
  Spacer,
  useColorMode,
} from '@chakra-ui/react';
import { useContext, useEffect, useState } from 'react';

import { FaSun, FaMoon, FaGithub, FaCircle } from 'react-icons/fa';
import { MdHelp } from 'react-icons/md';
import { SocketContext } from '../context/socket';
import SettingsModalButton from '../features/settings/SettingsModalButton';

const SiteHeader = () => {
  const { colorMode, toggleColorMode } = useColorMode();
  const socket = useContext(SocketContext);
  const [isConnected, setIsConnected] = useState<boolean>(socket.connected);

  useEffect(() => {
    socket.on('connect', () => {
      setIsConnected(true);
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
    });

    // socket.on('message', (data) => {
    //     console.log(data); // undefined
    // });

    // socket.on('progress', (data) => {
    //     const progress = Math.round(data * 100);
    //     console.log(`Progress: ${progress}%`);
    //     dispatch(setProgress(progress));
    // });

    // socket.on('image', (data) => {
    //     console.log(data); // undefined
    // });

    return () => {
      socket.off('connect');
      socket.off('disconnect');
      // socket.off('message');
      // socket.off('progress');
      // socket.off('image');
    };
  }, [socket]);

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

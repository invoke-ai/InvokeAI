import { Flex, Image } from '@chakra-ui/react';
import InvokeAILogoImage from 'assets/images/logo.png';

// This component loads before the theme so we cannot use theme tokens here

const Loading = () => {
  return (
    <Flex
      position="relative"
      width="100vw"
      height="100vh"
      alignItems="center"
      justifyContent="center"
      bg="#151519"
    >
      <Image src={InvokeAILogoImage} w="8rem" h="8rem" />
    </Flex>
  );
};

export default Loading;

import { Flex } from '@chakra-ui/react';
import { FaEyeSlash } from 'react-icons/fa';

const CurrentImageHidden = () => {
  return (
    <Flex
      sx={{
        w: 'full',
        h: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'absolute',
        color: 'base.400',
      }}
    >
      <FaEyeSlash fontSize="25vh" />
    </Flex>
  );
};

export default CurrentImageHidden;

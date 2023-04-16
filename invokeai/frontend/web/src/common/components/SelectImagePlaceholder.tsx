import { Flex, Icon } from '@chakra-ui/react';
import { FaImage } from 'react-icons/fa';

const SelectImagePlaceholder = () => {
  return (
    <Flex sx={{ alignItems: 'center', justifyContent: 'center' }}>
      <Icon color="base.400" boxSize={32} as={FaImage}></Icon>
    </Flex>
  );
};

export default SelectImagePlaceholder;

import { Flex, useDisclosure } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import ParamLoraList from './ParamLoraList';
import ParamLoraSelect from './ParamLoraSelect';

const ParamLoraCollapse = () => {
  const { isOpen, onToggle } = useDisclosure();

  return (
    <IAICollapse label="LoRAs" isOpen={isOpen} onToggle={onToggle}>
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ParamLoraSelect />
        <ParamLoraList />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamLoraCollapse);

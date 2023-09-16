import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import ParamIPAdapterFeatureToggle from './ParamIPAdapterFeatureToggle';
import ParamIPAdapterImage from './ParamIPAdapterImage';
import ParamIPAdapterModelSelect from './ParamIPAdapterModelSelect';
import ParamIPAdapterWeight from './ParamIPAdapterWeight';

const IPAdapterPanel = () => {
  return (
    <Flex
      sx={{
        flexDir: 'column',
        gap: 3,
        paddingInline: 3,
        paddingBlock: 2,
        paddingBottom: 5,
        borderRadius: 'base',
        position: 'relative',
        bg: 'base.250',
        _dark: {
          bg: 'base.750',
        },
      }}
    >
      <ParamIPAdapterFeatureToggle />
      <ParamIPAdapterImage />
      <ParamIPAdapterModelSelect />
      <ParamIPAdapterWeight />
    </Flex>
  );
};

export default memo(IPAdapterPanel);

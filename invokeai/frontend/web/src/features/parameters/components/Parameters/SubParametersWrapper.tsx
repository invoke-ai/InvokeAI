import { Flex, Text } from '@chakra-ui/react';
import { ReactNode, memo } from 'react';

type SubParameterWrapperProps = {
  children: ReactNode | ReactNode[];
  label?: string;
};

const SubParametersWrapper = (props: SubParameterWrapperProps) => (
  <Flex
    sx={{
      flexDir: 'column',
      gap: 2,
      bg: 'base.100',
      px: 4,
      pt: 2,
      pb: 4,
      borderRadius: 'base',
      _dark: {
        bg: 'base.750',
      },
    }}
  >
    {props.label && (
      <Text
        fontSize="sm"
        fontWeight="bold"
        sx={{ color: 'base.600', _dark: { color: 'base.300' } }}
      >
        {props.label}
      </Text>
    )}
    {props.children}
  </Flex>
);

SubParametersWrapper.displayName = 'SubSettingsWrapper';

export default memo(SubParametersWrapper);

import {
  As,
  Flex,
  FlexProps,
  Icon,
  IconProps,
  Spinner,
  SpinnerProps,
  useColorMode,
} from '@chakra-ui/react';
import { FaImage } from 'react-icons/fa';
import { mode } from 'theme/util/mode';

type Props = FlexProps & {
  spinnerProps?: SpinnerProps;
};

export const IAIImageLoadingFallback = (props: Props) => {
  const { spinnerProps, ...rest } = props;
  const { sx, ...restFlexProps } = rest;
  const { colorMode } = useColorMode();
  return (
    <Flex
      sx={{
        bg: mode('base.200', 'base.900')(colorMode),
        opacity: 0.7,
        w: 'full',
        h: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 'base',
        ...sx,
      }}
      {...restFlexProps}
    >
      <Spinner size="xl" {...spinnerProps} />
    </Flex>
  );
};

type IAINoImageFallbackProps = {
  flexProps?: FlexProps;
  iconProps?: IconProps;
  as?: As;
};

export const IAINoImageFallback = (props: IAINoImageFallbackProps) => {
  const { sx: flexSx, ...restFlexProps } = props.flexProps ?? { sx: {} };
  const { sx: iconSx, ...restIconProps } = props.iconProps ?? { sx: {} };
  const { colorMode } = useColorMode();

  return (
    <Flex
      sx={{
        bg: mode('base.200', 'base.900')(colorMode),
        opacity: 0.7,
        w: 'full',
        h: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 'base',
        ...flexSx,
      }}
      {...restFlexProps}
    >
      <Icon
        as={props.as ?? FaImage}
        sx={{ color: mode('base.700', 'base.500')(colorMode), ...iconSx }}
        {...restIconProps}
      />
    </Flex>
  );
};

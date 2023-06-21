import {
  As,
  Flex,
  FlexProps,
  Icon,
  IconProps,
  Spinner,
  SpinnerProps,
} from '@chakra-ui/react';
import { ReactElement } from 'react';
import { FaImage } from 'react-icons/fa';

type Props = FlexProps & {
  spinnerProps?: SpinnerProps;
};

export const IAIImageLoadingFallback = (props: Props) => {
  const { spinnerProps, ...rest } = props;
  const { sx, ...restFlexProps } = rest;
  return (
    <Flex
      sx={{
        bg: 'base.900',
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
  return (
    <Flex
      sx={{
        bg: 'base.900',
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
        sx={{ color: 'base.700', ...iconSx }}
        {...restIconProps}
      />
    </Flex>
  );
};

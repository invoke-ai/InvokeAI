import { Flex, FlexProps, Spinner, SpinnerProps } from '@chakra-ui/react';

type Props = FlexProps & {
  spinnerProps?: SpinnerProps;
};

export const IAIImageFallback = (props: Props) => {
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

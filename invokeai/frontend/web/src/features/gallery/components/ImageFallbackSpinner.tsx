import { Flex, Spinner, SpinnerProps } from '@chakra-ui/react';

type ImageFallbackSpinnerProps = SpinnerProps;

const ImageFallbackSpinner = (props: ImageFallbackSpinnerProps) => {
  const { size = 'xl', ...rest } = props;

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
      <Spinner size={size} {...rest} />
    </Flex>
  );
};

export default ImageFallbackSpinner;

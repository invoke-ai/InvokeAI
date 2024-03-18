import type { As, ChakraProps, FlexProps } from '@invoke-ai/ui-library';
import { Flex, Icon, Skeleton, Spinner, Text } from '@invoke-ai/ui-library';
import { memo, useMemo } from 'react';
import { PiImageBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

type Props = { image: ImageDTO | undefined };

export const IAILoadingImageFallback = memo((props: Props) => {
  if (props.image) {
    return (
      <Skeleton
        w={`${props.image.width}px`}
        h="auto"
        objectFit="contain"
        aspectRatio={`${props.image.width}/${props.image.height}`}
      />
    );
  }

  return (
    <Flex opacity={0.7} w="full" h="full" alignItems="center" justifyContent="center" borderRadius="base" bg="base.900">
      <Spinner size="xl" />
    </Flex>
  );
});
IAILoadingImageFallback.displayName = 'IAILoadingImageFallback';

type IAINoImageFallbackProps = FlexProps & {
  label?: string;
  icon?: As | null;
  boxSize?: ChakraProps['boxSize'];
};

export const IAINoContentFallback = memo((props: IAINoImageFallbackProps) => {
  const { icon = PiImageBold, boxSize = 16, sx, ...rest } = props;

  const styles = useMemo(
    () => ({
      w: 'full',
      h: 'full',
      alignItems: 'center',
      justifyContent: 'center',
      borderRadius: 'base',
      flexDir: 'column',
      gap: 2,
      userSelect: 'none',
      opacity: 0.7,
      color: 'base.500',
      ...sx,
    }),
    [sx]
  );

  return (
    <Flex sx={styles} {...rest}>
      {icon && <Icon as={icon} boxSize={boxSize} opacity={0.7} />}
      {props.label && (
        <Text textAlign="center" fontSize="md">
          {props.label}
        </Text>
      )}
    </Flex>
  );
});
IAINoContentFallback.displayName = 'IAINoContentFallback';

type IAINoImageFallbackWithSpinnerProps = FlexProps & {
  label?: string;
};

export const IAINoContentFallbackWithSpinner = memo((props: IAINoImageFallbackWithSpinnerProps) => {
  const { sx, ...rest } = props;
  const styles = useMemo(
    () => ({
      w: 'full',
      h: 'full',
      alignItems: 'center',
      justifyContent: 'center',
      borderRadius: 'base',
      flexDir: 'column',
      gap: 2,
      userSelect: 'none',
      opacity: 0.7,
      color: 'base.500',
      ...sx,
    }),
    [sx]
  );

  return (
    <Flex sx={styles} {...rest}>
      <Spinner size="xl" />
      {props.label && <Text textAlign="center">{props.label}</Text>}
    </Flex>
  );
});
IAINoContentFallbackWithSpinner.displayName = 'IAINoContentFallbackWithSpinner';

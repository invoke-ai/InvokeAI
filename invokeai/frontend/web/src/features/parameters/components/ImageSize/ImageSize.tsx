import { Flex } from '@chakra-ui/layout';
import { InvControlGroup } from 'common/components/InvControl/InvControlGroup';
import type { InvLabelProps } from 'common/components/InvControl/types';
import { AspectRatioPreview } from 'features/parameters/components/ImageSize/AspectRatioPreview';
import { AspectRatioSelect } from 'features/parameters/components/ImageSize/AspectRatioSelect';
import type { ImageSizeContextInnerValue } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { ImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { LockAspectRatioButton } from 'features/parameters/components/ImageSize/LockAspectRatioButton';
import { SetOptimalSizeButton } from 'features/parameters/components/ImageSize/SetOptimalSizeButton';
import { SwapDimensionsButton } from 'features/parameters/components/ImageSize/SwapDimensionsButton';
import type { ReactNode } from 'react';
import { memo } from 'react';

type ImageSizeProps = ImageSizeContextInnerValue & {
  widthComponent: ReactNode;
  heightComponent: ReactNode;
};

export const ImageSize = memo((props: ImageSizeProps) => {
  const { widthComponent, heightComponent, ...ctx } = props;
  return (
    <ImageSizeContext.Provider value={ctx}>
      <Flex gap={4} alignItems="center">
        <Flex gap={4} flexDirection="column" width="full">
          <InvControlGroup labelProps={labelProps}>
            <Flex gap={2} alignItems="center" borderBottomWidth="1px" paddingY="0.5rem">
              <AspectRatioSelect />
              <SwapDimensionsButton />
              <LockAspectRatioButton />
              <SetOptimalSizeButton />
            </Flex>
            {widthComponent}
            {heightComponent}
          </InvControlGroup>
        </Flex>
        <Flex w="108px" h="108px" flexShrink={0} flexGrow={0}>
          <AspectRatioPreview />
        </Flex>
      </Flex>
    </ImageSizeContext.Provider>
  );
});

ImageSize.displayName = 'ImageSize';

const labelProps: InvLabelProps = {
  minW: 14,
};

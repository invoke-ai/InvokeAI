import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Flex, FormControlGroup } from '@invoke-ai/ui-library';
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
          <FormControlGroup formLabelProps={formLabelProps}>
            <Flex gap={4}>
              <AspectRatioSelect />
              <SwapDimensionsButton />
              <LockAspectRatioButton />
              <SetOptimalSizeButton />
            </Flex>
            {widthComponent}
            {heightComponent}
          </FormControlGroup>
        </Flex>
        <Flex w="108px" h="108px" flexShrink={0} flexGrow={0}>
          <AspectRatioPreview />
        </Flex>
      </Flex>
    </ImageSizeContext.Provider>
  );
});

ImageSize.displayName = 'ImageSize';

const formLabelProps: FormLabelProps = {
  minW: 14,
};

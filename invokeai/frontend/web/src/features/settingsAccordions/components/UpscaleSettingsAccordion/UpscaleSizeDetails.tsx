import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useMemo } from 'react';

export const UpscaleSizeDetails = () => {
  const { upscaleInitialImage, scale } = useAppSelector((s) => s.upscale);

  const outputSizeText = useMemo(() => {
    if (upscaleInitialImage && scale) {
      return `Output image size: ${upscaleInitialImage.width * scale} x ${upscaleInitialImage.height * scale}`;
    }
  }, [upscaleInitialImage, scale]);

  if (!outputSizeText || !upscaleInitialImage) {
    return <></>;
  }

  return (
    <Flex direction="column">
      <Text variant="subtext" fontWeight="bold">
        Current image size: {upscaleInitialImage.width} x {upscaleInitialImage.height}
      </Text>
      <Text variant="subtext" fontWeight="bold">
        {outputSizeText}
      </Text>
    </Flex>
  );
};

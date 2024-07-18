import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useMemo } from 'react';

export const UpscaleSizeDetails = () => {
  const { upscaleInitialImage, upscaleModel } = useAppSelector((s) => s.upscale);

  const scaleFactor = useMemo(() => {
    if (upscaleModel) {
      const upscaleFactor = upscaleModel.name.match(/x(\d+)/);
      if (upscaleFactor && upscaleFactor[1]) {
        return parseInt(upscaleFactor[1], 10);
      }
    }
  }, [upscaleModel]);

  if (!upscaleInitialImage || !upscaleModel || !scaleFactor) {
    return <></>;
  }

  return (
    <Flex direction="column">
      <Text variant="subtext" fontWeight="bold">
        Current image size: {upscaleInitialImage.width} x {upscaleInitialImage.height}
      </Text>
      <Text variant="subtext" fontWeight="bold">
        Output image size: {upscaleInitialImage.width * scaleFactor} x {upscaleInitialImage.height * scaleFactor}
      </Text>
    </Flex>
  );
};

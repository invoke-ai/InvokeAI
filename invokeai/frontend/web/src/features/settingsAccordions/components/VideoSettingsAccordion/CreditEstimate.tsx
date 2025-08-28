import { Flex, Icon, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectVideoDuration, selectVideoModelConfig } from 'features/parameters/store/videoSlice';
import { useMemo } from 'react';
import { PiVideoBold } from 'react-icons/pi';

export const CreditEstimate = () => {
  const duration = useAppSelector(selectVideoDuration);
  const videoModel = useAppSelector(selectVideoModelConfig);

  const usagePerSecond = useMemo(() => {
    if (!videoModel) {
      return undefined;
    }
    const usageInfo = videoModel.usage_info;
    // Regex to parse "400" from "~400 credits"
    // Example: usageInfo = "~400 credits"
    if (typeof usageInfo === 'string') {
      const match = usageInfo.match(/~?(\d+)\s*credits?/i);
      if (match && match[1]) {
        return parseInt(match[1], 10);
      }
    }
    return undefined;
  }, [videoModel]);

  const intDuration = useMemo(() => {
    if (!duration) {
      return undefined;
    }
    return parseInt(duration, 10);
  }, [duration]);

  if (!usagePerSecond || !intDuration) {
    return null;
  }

  return (
    <Flex
      alignItems="center"
      layerStyle="first"
      py={2}
      px={6}
      borderRadius="base"
      justifyContent="center"
      gap={2}
      textAlign="center"
    >
      <Icon as={PiVideoBold} color="base.200" boxSize={7} />
      <Text color="base.200">
        Estimated usage for your current generation settings is {usagePerSecond * intDuration} credits
      </Text>
    </Flex>
  );
};

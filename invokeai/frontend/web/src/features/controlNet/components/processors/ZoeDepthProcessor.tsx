import { RequiredZoeDepthImageProcessorInvocation } from 'features/controlNet/store/types';
import { memo } from 'react';

type Props = {
  controlNetId: string;
  processorNode: RequiredZoeDepthImageProcessorInvocation;
};

const ZoeDepthProcessor = (props: Props) => {
  // Has no parameters?
  return null;
};

export default memo(ZoeDepthProcessor);

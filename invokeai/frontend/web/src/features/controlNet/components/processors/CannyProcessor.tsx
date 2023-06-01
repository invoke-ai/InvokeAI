import { Flex } from '@chakra-ui/react';
import IAISlider from 'common/components/IAISlider';
import { memo, useCallback, useState } from 'react';
import ControlNetProcessButton from './common/ControlNetProcessButton';
import { useAppDispatch } from 'app/store/storeHooks';
import { controlNetImageProcessed } from 'features/controlNet/store/actions';
import ControlNetResetProcessedImageButton from './common/ControlNetResetProcessedImageButton';
import { ControlNetProcessorProps } from '../ControlNetProcessorCollapse';
import { controlNetProcessedImageChanged } from 'features/controlNet/store/controlNetSlice';

export const CANNY_PROCESSOR = 'canny_image_processor';

const CannyProcessor = (props: ControlNetProcessorProps) => {
  const { controlNetId, image, type } = props;
  const dispatch = useAppDispatch();
  const [lowThreshold, setLowThreshold] = useState(100);
  const [highThreshold, setHighThreshold] = useState(200);

  const handleProcess = useCallback(() => {
    if (!image) {
      return;
    }

    dispatch(
      controlNetImageProcessed({
        controlNetId,
        processorNode: {
          id: CANNY_PROCESSOR,
          type: 'canny_image_processor',
          image: {
            image_name: image.image_name,
            image_origin: image.image_origin,
          },
          low_threshold: lowThreshold,
          high_threshold: highThreshold,
        },
      })
    );
  }, [controlNetId, dispatch, highThreshold, image, lowThreshold]);

  const handleReset = useCallback(() => {
    dispatch(
      controlNetProcessedImageChanged({
        controlNetId,
        processedControlImage: null,
      })
    );
  }, [controlNetId, dispatch]);

  return (
    <Flex sx={{ flexDirection: 'column', gap: 2 }}>
      <IAISlider
        label="Low Threshold"
        value={lowThreshold}
        onChange={setLowThreshold}
        min={0}
        max={255}
        withInput
      />
      <IAISlider
        label="High Threshold"
        value={highThreshold}
        onChange={setHighThreshold}
        min={0}
        max={255}
        withInput
      />
      <Flex sx={{ gap: 4 }}>
        <ControlNetProcessButton onClick={handleProcess} />
        <ControlNetResetProcessedImageButton onClick={handleReset} />
      </Flex>
    </Flex>
  );
};

export default memo(CannyProcessor);

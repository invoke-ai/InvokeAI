import { Flex } from '@chakra-ui/react';
import IAISlider from 'common/components/IAISlider';
import { useAppDispatch } from 'app/store/storeHooks';
import { controlNetImageProcessed } from 'features/controlNet/store/actions';
import { controlNetProcessedImageChanged } from 'features/controlNet/store/controlNetSlice';
import ControlNetProcessorButtons from './common/ControlNetProcessorButtons';
import { memo, useCallback, useState } from 'react';
import { ControlNetProcessorProps } from '../ControlNet';

export const CANNY_PROCESSOR = 'canny_image_processor';

const CannyProcessor = (props: ControlNetProcessorProps) => {
  const { controlNetId, controlImage, processedControlImage, type } = props;
  const dispatch = useAppDispatch();
  const [lowThreshold, setLowThreshold] = useState(100);
  const [highThreshold, setHighThreshold] = useState(200);

  const handleProcess = useCallback(() => {
    if (!controlImage) {
      return;
    }

    dispatch(
      controlNetImageProcessed({
        controlNetId,
        processorNode: {
          id: CANNY_PROCESSOR,
          type: 'canny_image_processor',
          image: {
            image_name: controlImage.image_name,
            image_origin: controlImage.image_origin,
          },
          low_threshold: lowThreshold,
          high_threshold: highThreshold,
        },
      })
    );
  }, [controlNetId, dispatch, highThreshold, controlImage, lowThreshold]);

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
      <ControlNetProcessorButtons
        handleProcess={handleProcess}
        isProcessDisabled={Boolean(!controlImage)}
        handleReset={handleReset}
        isResetDisabled={Boolean(!processedControlImage)}
      />
    </Flex>
  );
};

export default memo(CannyProcessor);

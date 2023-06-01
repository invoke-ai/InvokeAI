import { Flex } from '@chakra-ui/react';
import IAISlider from 'common/components/IAISlider';
import { memo, useCallback, useState } from 'react';
import ControlNetProcessButton from './common/ControlNetProcessButton';
import { useAppDispatch } from 'app/store/storeHooks';
import { controlNetImageProcessed } from 'features/controlNet/store/actions';
import { ImageDTO } from 'services/api';
import ControlNetProcessorImage from './common/ControlNetProcessorImage';
import { ControlNetProcessorProps } from '../ControlNet';

export const CANNY_PROCESSOR = 'canny_processor';

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
      <ControlNetProcessButton onClick={handleProcess} />
    </Flex>
  );
};

export default memo(CannyProcessor);

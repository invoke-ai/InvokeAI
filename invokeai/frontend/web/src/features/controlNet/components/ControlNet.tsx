import { memo, useCallback, useState } from 'react';
import { ControlNetProcessorNode } from '../store/types';
import { ImageDTO } from 'services/api';
import CannyProcessor from './processors/CannyProcessor';
import {
  ControlNet,
  ControlNetModel,
  controlNetBeginStepPctChanged,
  controlNetEndStepPctChanged,
  controlNetImageChanged,
  controlNetModelChanged,
  controlNetProcessedImageChanged,
  controlNetRemoved,
  controlNetToggled,
  controlNetWeightChanged,
  isControlNetImageProcessedToggled,
} from '../store/controlNetSlice';
import { useAppDispatch } from 'app/store/storeHooks';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import IAISlider from 'common/components/IAISlider';
import ParamControlNetIsEnabled from './parameters/ParamControlNetIsEnabled';
import ParamControlNetModel from './parameters/ParamControlNetModel';
import ParamControlNetWeight from './parameters/ParamControlNetWeight';
import ParamControlNetBeginStepPct from './parameters/ParamControlNetBeginStepPct';
import ParamControlNetEndStepPct from './parameters/ParamControlNetEndStepPct';
import { Flex, HStack, VStack } from '@chakra-ui/react';
import IAISelectableImage from './parameters/IAISelectableImage';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAISwitch from 'common/components/IAISwitch';
import ParamControlNetIsPreprocessed from './parameters/ParamControlNetIsPreprocessed';
import IAICollapse from 'common/components/IAICollapse';
import ControlNetProcessorCollapse from './ControlNetProcessorCollapse';

type ControlNetProps = {
  controlNet: ControlNet;
};

const ControlNet = (props: ControlNetProps) => {
  const {
    controlNetId,
    isEnabled,
    model,
    weight,
    beginStepPct,
    endStepPct,
    controlImage,
    isControlImageProcessed,
    processedControlImage,
  } = props.controlNet;
  const dispatch = useAppDispatch();

  const [processorType, setProcessorType] = useState<
    ControlNetProcessorNode['type']
  >('canny_image_processor');

  const handleControlImageChanged = useCallback(
    (controlImage: ImageDTO) => {
      dispatch(controlNetImageChanged({ controlNetId, controlImage }));
    },
    [controlNetId, dispatch]
  );

  const handleControlImageReset = useCallback(() => {
    dispatch(controlNetImageChanged({ controlNetId, controlImage: null }));
  }, [controlNetId, dispatch]);

  const handleControlNetRemoved = useCallback(() => {
    dispatch(controlNetRemoved(controlNetId));
  }, [controlNetId, dispatch]);

  const handleProcessedControlImageChanged = useCallback(
    (processedControlImage: ImageDTO | null) => {
      dispatch(
        controlNetProcessedImageChanged({
          controlNetId,
          processedControlImage,
        })
      );
    },
    [controlNetId, dispatch]
  );

  return (
    <Flex sx={{ flexDir: 'column', gap: 3, pb: 4 }}>
      <IAIButton onClick={handleControlNetRemoved}>Remove ControlNet</IAIButton>
      <IAISelectableImage
        image={processedControlImage || controlImage}
        onChange={handleControlImageChanged}
        onReset={handleControlImageReset}
        resetIconSize="sm"
      />
      <ControlNetProcessorCollapse
        controlNetId={controlNetId}
        image={controlImage}
      />
      <ParamControlNetModel controlNetId={controlNetId} model={model} />
      <ParamControlNetIsEnabled
        controlNetId={controlNetId}
        isEnabled={isEnabled}
      />
      <ParamControlNetWeight controlNetId={controlNetId} weight={weight} />
      <ParamControlNetBeginStepPct
        controlNetId={controlNetId}
        beginStepPct={beginStepPct}
      />
      <ParamControlNetEndStepPct
        controlNetId={controlNetId}
        endStepPct={endStepPct}
      />
    </Flex>
  );
};

export default memo(ControlNet);

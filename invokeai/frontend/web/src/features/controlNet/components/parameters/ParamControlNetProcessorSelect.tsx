import IAICustomSelect from 'common/components/IAICustomSelect';
import { memo, useCallback } from 'react';
import {
  ControlNetProcessorNode,
  ControlNetProcessorType,
} from '../../store/types';
import { controlNetProcessorTypeChanged } from '../../store/controlNetSlice';
import { useAppDispatch } from 'app/store/storeHooks';
import { CONTROLNET_PROCESSORS } from '../../store/constants';

type ParamControlNetProcessorSelectProps = {
  controlNetId: string;
  processorNode: ControlNetProcessorNode;
};

const CONTROLNET_PROCESSOR_TYPES = Object.keys(
  CONTROLNET_PROCESSORS
) as ControlNetProcessorType[];

const ParamControlNetProcessorSelect = (
  props: ParamControlNetProcessorSelectProps
) => {
  const { controlNetId, processorNode } = props;
  const dispatch = useAppDispatch();
  const handleProcessorTypeChanged = useCallback(
    (v: string | null | undefined) => {
      dispatch(
        controlNetProcessorTypeChanged({
          controlNetId,
          processorType: v as ControlNetProcessorType,
        })
      );
    },
    [controlNetId, dispatch]
  );
  return (
    <IAICustomSelect
      label="Processor"
      items={CONTROLNET_PROCESSOR_TYPES}
      selectedItem={processorNode.type ?? 'canny_image_processor'}
      setSelectedItem={handleProcessorTypeChanged}
      withCheckIcon
    />
  );
};

export default memo(ParamControlNetProcessorSelect);

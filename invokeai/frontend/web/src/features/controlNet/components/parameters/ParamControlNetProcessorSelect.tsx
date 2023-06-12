import IAICustomSelect, {
  IAICustomSelectOption,
} from 'common/components/IAICustomSelect';
import { memo, useCallback } from 'react';
import {
  ControlNetProcessorNode,
  ControlNetProcessorType,
} from '../../store/types';
import { controlNetProcessorTypeChanged } from '../../store/controlNetSlice';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CONTROLNET_PROCESSORS } from '../../store/constants';
import { map } from 'lodash-es';
import { isProcessingSelector } from 'features/system/store/systemSelectors';
import { useIsReadyToInvoke } from 'common/hooks/useIsReadyToInvoke';

type ParamControlNetProcessorSelectProps = {
  controlNetId: string;
  processorNode: ControlNetProcessorNode;
};

const CONTROLNET_PROCESSOR_TYPES: IAICustomSelectOption[] = map(
  CONTROLNET_PROCESSORS,
  (p) => ({
    value: p.type,
    label: p.label,
    tooltip: p.description,
  })
).sort((a, b) =>
  // sort 'none' to the top
  a.value === 'none'
    ? -1
    : b.value === 'none'
    ? 1
    : a.label.localeCompare(b.label)
);

const ParamControlNetProcessorSelect = (
  props: ParamControlNetProcessorSelectProps
) => {
  const { controlNetId, processorNode } = props;
  const dispatch = useAppDispatch();
  const isReady = useIsReadyToInvoke();

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
      value={processorNode.type ?? 'canny_image_processor'}
      data={CONTROLNET_PROCESSOR_TYPES}
      onChange={handleProcessorTypeChanged}
      withCheckIcon
      isDisabled={!isReady}
    />
  );
};

export default memo(ParamControlNetProcessorSelect);

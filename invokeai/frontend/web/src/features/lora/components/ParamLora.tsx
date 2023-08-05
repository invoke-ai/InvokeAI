import { Flex } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import IAISlider from 'common/components/IAISlider';
import { memo, useCallback } from 'react';
import { FaTrash } from 'react-icons/fa';
import {
  LoRA,
  loraRemoved,
  loraWeightChanged,
  loraWeightReset,
} from '../store/loraSlice';

type Props = {
  lora: LoRA;
};

const ParamLora = (props: Props) => {
  const dispatch = useAppDispatch();
  const { lora } = props;

  const handleChange = useCallback(
    (v: number) => {
      dispatch(loraWeightChanged({ id: lora.id, weight: v }));
    },
    [dispatch, lora.id]
  );

  const handleReset = useCallback(() => {
    dispatch(loraWeightReset(lora.id));
  }, [dispatch, lora.id]);

  const handleRemoveLora = useCallback(() => {
    dispatch(loraRemoved(lora.id));
  }, [dispatch, lora.id]);

  return (
    <Flex sx={{ gap: 2.5, alignItems: 'flex-end' }}>
      <IAISlider
        label={lora.model_name}
        value={lora.weight}
        onChange={handleChange}
        min={-1}
        max={2}
        step={0.01}
        withInput
        withReset
        handleReset={handleReset}
        withSliderMarks
        sliderMarks={[-1, 0, 1, 2]}
        sliderNumberInputProps={{ min: -50, max: 50 }}
      />
      <IAIIconButton
        size="sm"
        onClick={handleRemoveLora}
        tooltip="Remove LoRA"
        aria-label="Remove LoRA"
        icon={<FaTrash />}
        colorScheme="error"
      />
    </Flex>
  );
};

export default memo(ParamLora);

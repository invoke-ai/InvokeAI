import {
  Card,
  CardBody,
  CardHeader,
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  IconButton,
  Switch,
  Text,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import type { LoRA } from 'features/lora/store/loraSlice';
import { loraRemoved, loraWeightChanged } from 'features/lora/store/loraSlice';
import { memo, useCallback, useState } from 'react';
import { PiTrashSimpleBold } from 'react-icons/pi';

type LoRACardProps = {
  lora: LoRA;
};

export const LoRACard = memo((props: LoRACardProps) => {
  const dispatch = useAppDispatch();
  const { lora } = props;

  const [prevWeight, setPrevWeight] = useState(lora.weight);
  const [isEnabled, setIsEnabled] = useState(lora.weight !== 0);

  const handleChange = useCallback(
    (v: number) => {
      if (v !== 0) {
        setPrevWeight(v)
      }
      v === 0 ? setIsEnabled(false) : setIsEnabled(true);
      dispatch(loraWeightChanged({ id: lora.id, weight: v }));
    },
    [dispatch, lora.id]
  );

  const handleSetDisabled = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.checked) {
      handleChange(prevWeight);
    } else {
      setPrevWeight(lora.weight);
      handleChange(0);
    }
  }, [handleChange, lora.weight, prevWeight])

  const handleRemoveLora = useCallback(() => {
    dispatch(loraRemoved(lora.id));
  }, [dispatch, lora.id]);

  return (
    <Card variant="lora">
      <CardHeader>
        <Flex alignItems='center' justifyContent='space-between' width='100%' gap={2}>
          <Text noOfLines={1} wordBreak="break-all" color={isEnabled ? 'base.200' : 'base.500'}>
            {lora.model_name}
          </Text>
          <Flex alignItems='center' gap={2}>
            <Switch size='sm' onChange={handleSetDisabled} isChecked={isEnabled} />
            <IconButton
              aria-label="Remove LoRA"
              variant="ghost"
              size="sm"
              onClick={handleRemoveLora}
              icon={<PiTrashSimpleBold />}
            />
          </Flex>
        </Flex>
      </CardHeader>
      <CardBody>
        <CompositeSlider
          value={lora.weight}
          onChange={handleChange}
          min={-1}
          max={2}
          step={0.01}
          marks={marks}
          defaultValue={0.75}
        />
        <CompositeNumberInput
          value={lora.weight}
          onChange={handleChange}
          min={-5}
          max={5}
          step={0.01}
          w={20}
          flexShrink={0}
          defaultValue={0.75}
        />
      </CardBody>
    </Card>
  );
});

LoRACard.displayName = 'LoRACard';

const marks = [-1, 0, 1, 2];

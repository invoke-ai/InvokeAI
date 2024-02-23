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
import { loraIsEnabledChanged, loraRemoved, loraWeightChanged } from 'features/lora/store/loraSlice';
import { memo, useCallback } from 'react';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

type LoRACardProps = {
  lora: LoRA;
};

export const LoRACard = memo((props: LoRACardProps) => {
  const { lora } = props;
  const dispatch = useAppDispatch();
  const { data: loraConfig } = useGetModelConfigQuery(lora.model.key);

  const handleChange = useCallback(
    (v: number) => {
      dispatch(loraWeightChanged({ key: lora.model.key, weight: v }));
    },
    [dispatch, lora.model.key]
  );

  const handleSetLoraToggle = useCallback(() => {
    dispatch(loraIsEnabledChanged({ key: lora.model.key, isEnabled: !lora.isEnabled }));
  }, [dispatch, lora.model.key, lora.isEnabled]);

  const handleRemoveLora = useCallback(() => {
    dispatch(loraRemoved(lora.model.key));
  }, [dispatch, lora.model.key]);

  return (
    <Card variant="lora">
      <CardHeader>
        <Flex alignItems="center" justifyContent="space-between" width="100%" gap={2}>
          <Text noOfLines={1} wordBreak="break-all" color={lora.isEnabled ? 'base.200' : 'base.500'}>
            {loraConfig?.name ?? lora.model.key.substring(0, 8)}
          </Text>
          <Flex alignItems="center" gap={2}>
            <Switch size="sm" onChange={handleSetLoraToggle} isChecked={lora.isEnabled} />
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
          isDisabled={!lora.isEnabled}
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
          isDisabled={!lora.isEnabled}
        />
      </CardBody>
    </Card>
  );
});

LoRACard.displayName = 'LoRACard';

const marks = [-1, 0, 1, 2];

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
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  buildSelectLoRA,
  loraDeleted,
  loraIsEnabledChanged,
  loraWeightChanged,
} from 'features/controlLayers/store/lorasSlice';
import type { LoRA } from 'features/controlLayers/store/types';
import { DEFAULT_LORA_WEIGHT_CONFIG } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

export const LoRACard = memo((props: { id: string }) => {
  const selectLoRA = useMemo(() => buildSelectLoRA(props.id), [props.id]);
  const lora = useAppSelector(selectLoRA);

  if (!lora) {
    return null;
  }
  return <LoRAContent lora={lora} />;
});

LoRACard.displayName = 'LoRACard';

const LoRAContent = memo(({ lora }: { lora: LoRA }) => {
  const dispatch = useAppDispatch();
  const { data: loraConfig } = useGetModelConfigQuery(lora.model.key);

  const handleChange = useCallback(
    (v: number) => {
      dispatch(loraWeightChanged({ id: lora.id, weight: v }));
    },
    [dispatch, lora.id]
  );

  const handleSetLoraToggle = useCallback(() => {
    dispatch(loraIsEnabledChanged({ id: lora.id, isEnabled: !lora.isEnabled }));
  }, [dispatch, lora.id, lora.isEnabled]);

  const handleRemoveLora = useCallback(() => {
    dispatch(loraDeleted({ id: lora.id }));
  }, [dispatch, lora.id]);

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
      <InformationalPopover feature="loraWeight">
        <CardBody>
          <CompositeSlider
            value={lora.weight}
            onChange={handleChange}
            min={DEFAULT_LORA_WEIGHT_CONFIG.sliderMin}
            max={DEFAULT_LORA_WEIGHT_CONFIG.sliderMax}
            step={DEFAULT_LORA_WEIGHT_CONFIG.fineStep}
            marks={DEFAULT_LORA_WEIGHT_CONFIG.marks.slice()}
            defaultValue={DEFAULT_LORA_WEIGHT_CONFIG.initial}
            isDisabled={!lora.isEnabled}
          />
          <CompositeNumberInput
            value={lora.weight}
            onChange={handleChange}
            min={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin}
            max={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax}
            step={DEFAULT_LORA_WEIGHT_CONFIG.fineStep}
            w={20}
            flexShrink={0}
            defaultValue={DEFAULT_LORA_WEIGHT_CONFIG.initial}
            isDisabled={!lora.isEnabled}
          />
        </CardBody>
      </InformationalPopover>
    </Card>
  );
});

LoRAContent.displayName = 'LoRAContent';

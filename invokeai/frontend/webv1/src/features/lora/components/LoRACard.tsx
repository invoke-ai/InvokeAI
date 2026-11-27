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
  DEFAULT_LORA_WEIGHT_CONFIG,
  loraDeleted,
  loraIsEnabledChanged,
  loraWeightChanged,
} from 'features/controlLayers/store/lorasSlice';
import type { LoRA } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';
import { isLoRAModelConfig } from 'services/api/types';

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
  const { t } = useTranslation();
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

  const loraDefaults = loraConfig && isLoRAModelConfig(loraConfig) ? loraConfig.default_settings : null;
  const sliderMin = loraDefaults?.weight_min ?? DEFAULT_LORA_WEIGHT_CONFIG.sliderMin;
  const sliderMax = loraDefaults?.weight_max ?? DEFAULT_LORA_WEIGHT_CONFIG.sliderMax;
  const numberInputMin = Math.min(sliderMin, DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin);
  const numberInputMax = Math.max(sliderMax, DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax);

  const marks = useMemo(() => {
    if (sliderMin >= sliderMax) {
      return [sliderMin, sliderMax];
    }
    const mid = (sliderMin + sliderMax) / 2;
    return [sliderMin, mid, sliderMax];
  }, [sliderMin, sliderMax]);

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
              aria-label={t('lora.removeLoRA')}
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
            min={sliderMin}
            max={sliderMax}
            step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
            fineStep={DEFAULT_LORA_WEIGHT_CONFIG.fineStep}
            marks={marks}
            defaultValue={DEFAULT_LORA_WEIGHT_CONFIG.initial}
            isDisabled={!lora.isEnabled}
          />
          <CompositeNumberInput
            value={lora.weight}
            onChange={handleChange}
            min={numberInputMin}
            max={numberInputMax}
            step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
            fineStep={DEFAULT_LORA_WEIGHT_CONFIG.fineStep}
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

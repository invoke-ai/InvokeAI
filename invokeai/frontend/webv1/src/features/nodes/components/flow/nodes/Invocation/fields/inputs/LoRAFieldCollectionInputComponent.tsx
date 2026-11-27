import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { CompositeNumberInput, Divider, Flex, Grid, GridItem, IconButton, Text } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppStore } from 'app/store/storeHooks';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { DEFAULT_LORA_WEIGHT_CONFIG } from 'features/controlLayers/store/lorasSlice';
import { useInputFieldIsInvalid } from 'features/nodes/hooks/useInputFieldIsInvalid';
import { fieldLoRACollectionValueChanged } from 'features/nodes/store/nodesSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type {
  LoRAFieldCollectionInputInstance,
  LoRAFieldCollectionInputTemplate,
  LoRAFieldValue,
} from 'features/nodes/types/field';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { useLoRAModels } from 'services/api/hooks/modelsByType';
import type { LoRAModelConfig } from 'services/api/types';

import type { FieldComponentProps } from './types';

const overlayscrollbarsOptions = getOverlayScrollbarsParams({}).options;

const sx = {
  borderWidth: 1,
  '&[data-error=true]': {
    borderColor: 'error.500',
    borderStyle: 'solid',
  },
} satisfies SystemStyleObject;

export const LoRAFieldCollectionInputComponent = memo(
  (props: FieldComponentProps<LoRAFieldCollectionInputInstance, LoRAFieldCollectionInputTemplate>) => {
    const { nodeId, field, fieldTemplate } = props;
    const store = useAppStore();
    const { t } = useTranslation();

    const isInvalid = useInputFieldIsInvalid(field.name);
    const [allLoRAModels, { isLoading }] = useLoRAModels();

    const value = useMemo(() => field.value ?? EMPTY_ARRAY, [field.value]);

    // Filter the picker to LoRAs compatible with the node's base model(s). The collection loaders
    // stamp `ui_model_base` so each node only offers LoRAs it can actually apply.
    const compatibleLoRAs = useMemo(() => {
      const bases = fieldTemplate.ui_model_base;
      if (!bases || bases.length === 0) {
        return allLoRAModels;
      }
      return allLoRAModels.filter((model) => bases.includes(model.base));
    }, [allLoRAModels, fieldTemplate.ui_model_base]);

    const onChangeValue = useCallback(
      (newValue: LoRAFieldValue[]) => {
        store.dispatch(fieldLoRACollectionValueChanged({ nodeId, fieldName: field.name, value: newValue }));
      },
      [field.name, nodeId, store]
    );

    const onAddLoRA = useCallback(
      (model: LoRAModelConfig) => {
        const newItem: LoRAFieldValue = {
          lora: zModelIdentifierField.parse(model),
          weight: model.default_settings?.weight ?? DEFAULT_LORA_WEIGHT_CONFIG.initial,
        };
        onChangeValue([...value, newItem]);
      },
      [onChangeValue, value]
    );

    const onRemoveLoRA = useCallback(
      (index: number) => {
        const newValue = [...value];
        newValue.splice(index, 1);
        onChangeValue(newValue);
      },
      [onChangeValue, value]
    );

    const onChangeWeight = useCallback(
      (index: number, weight: number) => {
        const newValue = [...value];
        const item = newValue[index];
        if (!item) {
          return;
        }
        newValue[index] = { ...item, weight };
        onChangeValue(newValue);
      },
      [onChangeValue, value]
    );

    // Disable LoRAs that have already been added so the user can't add duplicates (the backend
    // dedupes by key anyway, but this is clearer in the UI).
    const addedKeys = useMemo(() => new Set(value.map((item) => item.lora.key)), [value]);
    const getIsOptionDisabled = useCallback((model: LoRAModelConfig) => addedKeys.has(model.key), [addedKeys]);

    const placeholder = useMemo(() => {
      if (isLoading) {
        return t('common.loading');
      }
      if (compatibleLoRAs.length === 0) {
        return t('models.noCompatibleLoRAs');
      }
      return t('models.addLora');
    }, [compatibleLoRAs.length, isLoading, t]);

    return (
      <Flex
        className={NO_DRAG_CLASS}
        position="relative"
        w="full"
        h="auto"
        maxH={64}
        alignItems="stretch"
        justifyContent="center"
        p={1}
        sx={sx}
        data-error={isInvalid}
        borderRadius="base"
        flexDir="column"
        gap={1}
      >
        <ModelPicker
          pickerId={`lora-collection-${nodeId}-${field.name}`}
          modelConfigs={compatibleLoRAs}
          selectedModelConfig={undefined}
          onChange={onAddLoRA}
          getIsOptionDisabled={getIsOptionDisabled}
          grouped={false}
          allowEmpty
          placeholder={placeholder}
          noOptionsText={t('models.noCompatibleLoRAs')}
        />
        {value.length > 0 && (
          <>
            <Divider />
            <OverlayScrollbarsComponent
              className={NO_WHEEL_CLASS}
              defer
              style={overlayScrollbarsStyles}
              options={overlayscrollbarsOptions}
            >
              <Grid gap={1} gridTemplateColumns="1fr auto auto" alignItems="center">
                {value.map((item, index) => (
                  <LoRAListItemContent
                    key={`${item.lora.key}-${index}`}
                    item={item}
                    index={index}
                    onRemoveLoRA={onRemoveLoRA}
                    onChangeWeight={onChangeWeight}
                  />
                ))}
              </Grid>
            </OverlayScrollbarsComponent>
          </>
        )}
      </Flex>
    );
  }
);

LoRAFieldCollectionInputComponent.displayName = 'LoRAFieldCollectionInputComponent';

type LoRAListItemContentProps = {
  item: LoRAFieldValue;
  index: number;
  onRemoveLoRA: (index: number) => void;
  onChangeWeight: (index: number, weight: number) => void;
};

const LoRAListItemContent = memo(({ item, index, onRemoveLoRA, onChangeWeight }: LoRAListItemContentProps) => {
  const { t } = useTranslation();

  const onClickRemove = useCallback(() => {
    onRemoveLoRA(index);
  }, [index, onRemoveLoRA]);

  const onChange = useCallback(
    (weight: number) => {
      onChangeWeight(index, weight);
    },
    [index, onChangeWeight]
  );

  return (
    <>
      <GridItem minW={0}>
        <Text noOfLines={1} title={item.lora.name}>
          {item.lora.name}
        </Text>
      </GridItem>
      <GridItem>
        <CompositeNumberInput
          onChange={onChange}
          value={item.weight}
          min={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin}
          max={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax}
          step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
          fineStep={DEFAULT_LORA_WEIGHT_CONFIG.fineStep}
          className={NO_DRAG_CLASS}
          w={20}
          allowMath
        />
      </GridItem>
      <GridItem>
        <IconButton
          tabIndex={-1}
          size="sm"
          variant="link"
          alignSelf="stretch"
          onClick={onClickRemove}
          icon={<PiXBold />}
          aria-label={t('common.delete')}
        />
      </GridItem>
    </>
  );
});
LoRAListItemContent.displayName = 'LoRAListItemContent';

import { CompositeNumberInput, Flex, FormControl, FormLabel, Select, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { roundDownToMultiple, roundUpToMultiple } from 'common/util/roundDownToMultiple';
import { useIntegerField } from 'features/nodes/components/flow/nodes/Invocation/fields/IntegerField/useIntegerField';
import { useInputFieldInstance } from 'features/nodes/hooks/useInputFieldInstance';
import { fieldIntegerValueChanged } from 'features/nodes/store/nodesSlice';
import { formElementNodeFieldDataChanged } from 'features/nodes/store/workflowSlice';
import type { IntegerFieldInputInstance, IntegerFieldInputTemplate } from 'features/nodes/types/field';
import type { NodeFieldIntegerSettings } from 'features/nodes/types/workflow';
import { zNumberComponent } from 'features/nodes/types/workflow';
import { constrainNumber } from 'features/nodes/util/constrainNumber';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
  config: NodeFieldIntegerSettings;
  nodeId: string;
  fieldName: string;
  fieldTemplate: IntegerFieldInputTemplate;
};

export const NodeFieldElementIntegerSettings = memo(({ id, config, nodeId, fieldName, fieldTemplate }: Props) => {
  return (
    <>
      <SettingComponent id={id} config={config} nodeId={nodeId} fieldName={fieldName} fieldTemplate={fieldTemplate} />
      <SettingMin id={id} config={config} nodeId={nodeId} fieldName={fieldName} fieldTemplate={fieldTemplate} />
      <SettingMax id={id} config={config} nodeId={nodeId} fieldName={fieldName} fieldTemplate={fieldTemplate} />
    </>
  );
});
NodeFieldElementIntegerSettings.displayName = 'NodeFieldElementIntegerSettings';

const SettingComponent = memo(({ id, config }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onChangeComponent = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const newConfig: NodeFieldIntegerSettings = {
        ...config,
        component: zNumberComponent.parse(e.target.value),
      };
      dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));
    },
    [config, dispatch, id]
  );

  return (
    <FormControl orientation="vertical">
      <FormLabel flex={1}>{t('workflows.builder.component')}</FormLabel>
      <Select value={config.component} onChange={onChangeComponent} size="sm">
        <option value="number-input">{t('workflows.builder.numberInput')}</option>
        <option value="slider">{t('workflows.builder.slider')}</option>
        <option value="number-input-and-slider">{t('workflows.builder.both')}</option>
      </Select>
    </FormControl>
  );
});
SettingComponent.displayName = 'SettingComponent';

const SettingMin = memo(({ id, config, nodeId, fieldName, fieldTemplate }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const field = useInputFieldInstance<IntegerFieldInputInstance>(nodeId, fieldName);

  const integerField = useIntegerField(nodeId, fieldName, fieldTemplate);

  const onToggleSetting = useCallback(() => {
    const newConfig: NodeFieldIntegerSettings = {
      ...config,
      min: config.min !== undefined ? undefined : integerField.min,
    };

    dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));
  }, [config, dispatch, integerField.min, id]);

  const onChange = useCallback(
    (min: number) => {
      const newConfig: NodeFieldIntegerSettings = {
        ...config,
        min,
      };
      dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));

      // We may need to update the value if it is outside the new min/max range
      const constrained = constrainNumber(field.value, integerField, newConfig);
      if (field.value !== constrained) {
        dispatch(fieldIntegerValueChanged({ nodeId, fieldName, value: constrained }));
      }
    },
    [config, dispatch, id, field, integerField, nodeId, fieldName]
  );

  const constraintMin = useMemo(
    () => roundUpToMultiple(integerField.min, integerField.step),
    [integerField.min, integerField.step]
  );

  const constraintMax = useMemo(
    () => (config.max ?? integerField.max) - integerField.step,
    [config.max, integerField.max, integerField.step]
  );

  return (
    <FormControl orientation="vertical">
      <Flex justifyContent="space-between" w="full" alignItems="center">
        <FormLabel m={0}>{t('workflows.builder.minimum')}</FormLabel>
        <Switch isChecked={config.min !== undefined} onChange={onToggleSetting} size="sm" />
      </Flex>
      <CompositeNumberInput
        w="full"
        isDisabled={config.min === undefined}
        value={config.min ?? (`${integerField.min} (inherited)` as unknown as number)}
        onChange={onChange}
        min={constraintMin}
        max={constraintMax}
        step={integerField.step}
      />
    </FormControl>
  );
});
SettingMin.displayName = 'SettingMin';

const SettingMax = memo(({ id, config, nodeId, fieldName, fieldTemplate }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const field = useInputFieldInstance<IntegerFieldInputInstance>(nodeId, fieldName);

  const integerField = useIntegerField(nodeId, fieldName, fieldTemplate);

  const onToggleSetting = useCallback(() => {
    const newConfig: NodeFieldIntegerSettings = {
      ...config,
      max: config.max !== undefined ? undefined : integerField.max,
    };
    dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));
  }, [config, dispatch, integerField.max, id]);

  const onChange = useCallback(
    (max: number) => {
      const newConfig: NodeFieldIntegerSettings = {
        ...config,
        max,
      };

      dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));

      // We may need to update the value if it is outside the new min/max range
      const constrained = constrainNumber(field.value, integerField, newConfig);
      if (field.value !== constrained) {
        dispatch(fieldIntegerValueChanged({ nodeId, fieldName, value: constrained }));
      }
    },
    [config, dispatch, field.value, fieldName, integerField, id, nodeId]
  );

  const constraintMin = useMemo(
    () => (config.min ?? integerField.min) + integerField.step,
    [config.min, integerField.min, integerField.step]
  );

  const constraintMax = useMemo(
    () => roundDownToMultiple(integerField.max, integerField.step),
    [integerField.max, integerField.step]
  );

  return (
    <FormControl orientation="vertical">
      <Flex justifyContent="space-between" w="full" alignItems="center">
        <FormLabel m={0}>{t('workflows.builder.maximum')}</FormLabel>
        <Switch isChecked={config.max !== undefined} onChange={onToggleSetting} size="sm" />
      </Flex>
      <CompositeNumberInput
        w="full"
        isDisabled={config.max === undefined}
        value={config.max ?? (`${integerField.max} (inherited)` as unknown as number)}
        onChange={onChange}
        min={constraintMin}
        max={constraintMax}
        step={integerField.step}
      />
    </FormControl>
  );
});
SettingMax.displayName = 'SettingMax';

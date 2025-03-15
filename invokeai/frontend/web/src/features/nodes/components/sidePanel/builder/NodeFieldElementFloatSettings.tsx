import { CompositeNumberInput, Flex, FormControl, FormLabel, Select, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { roundDownToMultiple, roundUpToMultiple } from 'common/util/roundDownToMultiple';
import { useFloatField } from 'features/nodes/components/flow/nodes/Invocation/fields/FloatField/useFloatField';
import { useInputFieldInstance } from 'features/nodes/hooks/useInputFieldInstance';
import { fieldFloatValueChanged } from 'features/nodes/store/nodesSlice';
import { formElementNodeFieldDataChanged } from 'features/nodes/store/workflowSlice';
import type { FloatFieldInputInstance, FloatFieldInputTemplate } from 'features/nodes/types/field';
import { type NodeFieldFloatSettings, zNumberComponent } from 'features/nodes/types/workflow';
import { constrainNumber } from 'features/nodes/util/constrainNumber';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
  config: NodeFieldFloatSettings;
  nodeId: string;
  fieldName: string;
  fieldTemplate: FloatFieldInputTemplate;
};

export const NodeFieldElementFloatSettings = memo(({ id, config, nodeId, fieldName, fieldTemplate }: Props) => {
  return (
    <>
      <SettingComponent id={id} config={config} nodeId={nodeId} fieldName={fieldName} fieldTemplate={fieldTemplate} />
      <SettingMin id={id} config={config} nodeId={nodeId} fieldName={fieldName} fieldTemplate={fieldTemplate} />
      <SettingMax id={id} config={config} nodeId={nodeId} fieldName={fieldName} fieldTemplate={fieldTemplate} />
    </>
  );
});
NodeFieldElementFloatSettings.displayName = 'NodeFieldElementFloatSettings';

const SettingComponent = memo(({ id, config }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onChangeComponent = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const newConfig: NodeFieldFloatSettings = {
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
  const field = useInputFieldInstance<FloatFieldInputInstance>(nodeId, fieldName);

  const floatField = useFloatField(nodeId, fieldName, fieldTemplate);

  const onToggleSetting = useCallback(() => {
    const newConfig: NodeFieldFloatSettings = {
      ...config,
      min: config.min !== undefined ? undefined : floatField.min,
    };
    dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));
  }, [config, dispatch, floatField, id]);

  const onChange = useCallback(
    (min: number) => {
      const newConfig: NodeFieldFloatSettings = {
        ...config,
        min,
      };
      dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));

      // We may need to update the value if it is outside the new min/max range
      const constrained = constrainNumber(field.value, floatField, newConfig);
      if (field.value !== constrained) {
        dispatch(fieldFloatValueChanged({ nodeId, fieldName, value: constrained }));
      }
    },
    [config, dispatch, field.value, fieldName, floatField, id, nodeId]
  );

  const constraintMin = useMemo(
    () => roundUpToMultiple(floatField.min, floatField.step),
    [floatField.min, floatField.step]
  );

  const constraintMax = useMemo(
    () => (config.max ?? floatField.max) - floatField.step,
    [config.max, floatField.max, floatField.step]
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
        value={config.min === undefined ? (`${floatField.min} (inherited)` as unknown as number) : config.min}
        onChange={onChange}
        min={constraintMin}
        max={constraintMax}
        step={floatField.step}
      />
    </FormControl>
  );
});
SettingMin.displayName = 'SettingMin';

const SettingMax = memo(({ id, config, nodeId, fieldName, fieldTemplate }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const field = useInputFieldInstance<FloatFieldInputInstance>(nodeId, fieldName);

  const floatField = useFloatField(nodeId, fieldName, fieldTemplate);

  const onToggleSetting = useCallback(() => {
    const newConfig: NodeFieldFloatSettings = {
      ...config,
      max: config.max !== undefined ? undefined : floatField.max,
    };
    dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));
  }, [config, dispatch, floatField, id]);

  const onChange = useCallback(
    (max: number) => {
      const newConfig: NodeFieldFloatSettings = {
        ...config,
        max,
      };
      dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));

      // We may need to update the value if it is outside the new min/max range
      const constrained = constrainNumber(field.value, floatField, newConfig);
      if (field.value !== constrained) {
        dispatch(fieldFloatValueChanged({ nodeId, fieldName, value: constrained }));
      }
    },
    [config, dispatch, field.value, fieldName, floatField, id, nodeId]
  );

  const constraintMin = useMemo(
    () => (config.min ?? floatField.min) + floatField.step,
    [config.min, floatField.min, floatField.step]
  );

  const constraintMax = useMemo(
    () => roundDownToMultiple(floatField.max, floatField.step),
    [floatField.max, floatField.step]
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
        value={config.max === undefined ? (`${floatField.max} (inherited)` as unknown as number) : config.max}
        onChange={onChange}
        min={constraintMin}
        max={constraintMax}
        step={floatField.step}
      />
    </FormControl>
  );
});
SettingMax.displayName = 'SettingMax';

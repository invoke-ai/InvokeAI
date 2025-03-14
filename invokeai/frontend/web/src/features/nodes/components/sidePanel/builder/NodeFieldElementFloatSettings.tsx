import { CompositeNumberInput, Flex, FormControl, FormLabel, Select, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useFloatField } from 'features/nodes/components/flow/nodes/Invocation/fields/FloatField/useFloatField';
import { useInputFieldInstance } from 'features/nodes/hooks/useInputFieldInstance';
import { fieldFloatValueChanged } from 'features/nodes/store/nodesSlice';
import { formElementNodeFieldDataChanged } from 'features/nodes/store/workflowSlice';
import type { FloatFieldInputInstance, FloatFieldInputTemplate } from 'features/nodes/types/field';
import { type NodeFieldFloatSettings, zNumberComponent } from 'features/nodes/types/workflow';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
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
  }, [config, dispatch, floatField.min, id]);

  const onChange = useCallback(
    (v: number) => {
      const newConfig: NodeFieldFloatSettings = {
        ...config,
        min: v,
      };
      dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));
      const constrained = constrain(v, floatField, newConfig);
      if (field.value !== constrained) {
        dispatch(fieldFloatValueChanged({ nodeId, fieldName, value: v }));
      }
    },
    [config, dispatch, field.value, fieldName, floatField, id, nodeId]
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
        min={floatField.min}
        max={(config.max ?? floatField.max) - 0.1}
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
  }, [config, dispatch, floatField.max, id]);

  const onChange = useCallback(
    (v: number) => {
      const newConfig: NodeFieldFloatSettings = {
        ...config,
        max: v,
      };
      dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newConfig } }));
      const constrained = constrain(v, floatField, newConfig);
      if (field.value !== constrained) {
        dispatch(fieldFloatValueChanged({ nodeId, fieldName, value: v }));
      }
    },
    [config, dispatch, field.value, fieldName, floatField, id, nodeId]
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
        min={(config.min ?? floatField.min) + 0.1}
        max={floatField.max}
      />
    </FormControl>
  );
});
SettingMax.displayName = 'SettingMax';

const constrain = (v: number, fieldSettings: ReturnType<typeof useFloatField>, overrides: NodeFieldFloatSettings) => {
  const min = overrides.min ?? fieldSettings.min;
  const max = overrides.max ?? fieldSettings.max;
  const step = overrides.step ?? fieldSettings.step;

  const _v = Math.min(max, Math.max(min, v));
  const _diff = _v - min;
  const _steps = Math.round(_diff / step);
  return min + _steps * step;
};

import { CompositeNumberInput, Flex, FormControl, FormLabel, Select, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { roundDownToMultiple, roundUpToMultiple } from 'common/util/roundDownToMultiple';
import { useFloatField } from 'features/nodes/components/flow/nodes/Invocation/fields/FloatField/useFloatField';
import { useInputFieldInstance } from 'features/nodes/hooks/useInputFieldInstance';
import { fieldFloatValueChanged, formElementNodeFieldDataChanged } from 'features/nodes/store/nodesSlice';
import type { FloatFieldInputInstance, FloatFieldInputTemplate } from 'features/nodes/types/field';
import { type NodeFieldFloatSettings, zNumberComponent } from 'features/nodes/types/workflow';
import { constrainNumber } from 'features/nodes/util/constrainNumber';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
  settings: NodeFieldFloatSettings;
  nodeId: string;
  fieldName: string;
  fieldTemplate: FloatFieldInputTemplate;
};

export const NodeFieldElementFloatSettings = memo(({ id, settings, nodeId, fieldName, fieldTemplate }: Props) => {
  return (
    <>
      <SettingComponent
        id={id}
        settings={settings}
        nodeId={nodeId}
        fieldName={fieldName}
        fieldTemplate={fieldTemplate}
      />
      <SettingMin id={id} settings={settings} nodeId={nodeId} fieldName={fieldName} fieldTemplate={fieldTemplate} />
      <SettingMax id={id} settings={settings} nodeId={nodeId} fieldName={fieldName} fieldTemplate={fieldTemplate} />
    </>
  );
});
NodeFieldElementFloatSettings.displayName = 'NodeFieldElementFloatSettings';

const SettingComponent = memo(({ id, settings }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onChangeComponent = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const newSettings: NodeFieldFloatSettings = {
        ...settings,
        component: zNumberComponent.parse(e.target.value),
      };
      dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newSettings } }));
    },
    [settings, dispatch, id]
  );

  return (
    <FormControl orientation="vertical">
      <FormLabel flex={1}>{t('workflows.builder.component')}</FormLabel>
      <Select value={settings.component} onChange={onChangeComponent} size="sm">
        <option value="number-input">{t('workflows.builder.numberInput')}</option>
        <option value="slider">{t('workflows.builder.slider')}</option>
        <option value="number-input-and-slider">{t('workflows.builder.both')}</option>
      </Select>
    </FormControl>
  );
});
SettingComponent.displayName = 'SettingComponent';

const SettingMin = memo(({ id, settings, nodeId, fieldName, fieldTemplate }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const field = useInputFieldInstance<FloatFieldInputInstance>(nodeId, fieldName);

  const floatField = useFloatField(nodeId, fieldName, fieldTemplate);

  const onToggleOverride = useCallback(() => {
    const newSettings: NodeFieldFloatSettings = {
      ...settings,
      min: settings.min !== undefined ? undefined : floatField.min,
    };
    dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newSettings } }));
  }, [settings, dispatch, floatField, id]);

  const onChange = useCallback(
    (min: number) => {
      const newSettings: NodeFieldFloatSettings = {
        ...settings,
        min,
      };
      dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newSettings } }));

      // We may need to update the value if it is outside the new min/max range
      const constrained = constrainNumber(field.value, floatField, newSettings);
      if (field.value !== constrained) {
        dispatch(fieldFloatValueChanged({ nodeId, fieldName, value: constrained }));
      }
    },
    [settings, dispatch, field.value, fieldName, floatField, id, nodeId]
  );

  const constraintMin = useMemo(
    () => roundUpToMultiple(floatField.min, floatField.step),
    [floatField.min, floatField.step]
  );

  const constraintMax = useMemo(
    () => (settings.max ?? floatField.max) - floatField.step,
    [settings.max, floatField.max, floatField.step]
  );

  return (
    <FormControl orientation="vertical">
      <Flex justifyContent="space-between" w="full" alignItems="center">
        <FormLabel m={0}>{t('workflows.builder.minimum')}</FormLabel>
        <Switch isChecked={settings.min !== undefined} onChange={onToggleOverride} size="sm" />
      </Flex>
      <CompositeNumberInput
        w="full"
        isDisabled={settings.min === undefined}
        value={settings.min === undefined ? (`${floatField.min} (inherited)` as unknown as number) : settings.min}
        onChange={onChange}
        min={constraintMin}
        max={constraintMax}
        step={floatField.step}
      />
    </FormControl>
  );
});
SettingMin.displayName = 'SettingMin';

const SettingMax = memo(({ id, settings, nodeId, fieldName, fieldTemplate }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const field = useInputFieldInstance<FloatFieldInputInstance>(nodeId, fieldName);

  const floatField = useFloatField(nodeId, fieldName, fieldTemplate);

  const onToggleOverride = useCallback(() => {
    const newSettings: NodeFieldFloatSettings = {
      ...settings,
      max: settings.max !== undefined ? undefined : floatField.max,
    };
    dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newSettings } }));
  }, [settings, dispatch, floatField, id]);

  const onChange = useCallback(
    (max: number) => {
      const newSettings: NodeFieldFloatSettings = {
        ...settings,
        max,
      };
      dispatch(formElementNodeFieldDataChanged({ id, changes: { settings: newSettings } }));

      // We may need to update the value if it is outside the new min/max range
      const constrained = constrainNumber(field.value, floatField, newSettings);
      if (field.value !== constrained) {
        dispatch(fieldFloatValueChanged({ nodeId, fieldName, value: constrained }));
      }
    },
    [settings, dispatch, field.value, fieldName, floatField, id, nodeId]
  );

  const constraintMin = useMemo(
    () => (settings.min ?? floatField.min) + floatField.step,
    [settings.min, floatField.min, floatField.step]
  );

  const constraintMax = useMemo(
    () => roundDownToMultiple(floatField.max, floatField.step),
    [floatField.max, floatField.step]
  );

  return (
    <FormControl orientation="vertical">
      <Flex justifyContent="space-between" w="full" alignItems="center">
        <FormLabel m={0}>{t('workflows.builder.maximum')}</FormLabel>
        <Switch isChecked={settings.max !== undefined} onChange={onToggleOverride} size="sm" />
      </Flex>
      <CompositeNumberInput
        w="full"
        isDisabled={settings.max === undefined}
        value={settings.max === undefined ? (`${floatField.max} (inherited)` as unknown as number) : settings.max}
        onChange={onChange}
        min={constraintMin}
        max={constraintMax}
        step={floatField.step}
      />
    </FormControl>
  );
});
SettingMax.displayName = 'SettingMax';

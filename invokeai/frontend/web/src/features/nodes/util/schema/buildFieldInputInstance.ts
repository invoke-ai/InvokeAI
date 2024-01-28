import type { FieldInputInstance, FieldInputTemplate, FieldValue, StatefulFieldType } from 'features/nodes/types/field';
import { get } from 'lodash-es';

const FIELD_VALUE_FALLBACK_MAP: Record<StatefulFieldType['name'], FieldValue> = {
  EnumField: '',
  BoardField: undefined,
  BooleanField: false,
  ColorField: { r: 0, g: 0, b: 0, a: 1 },
  FloatField: 0,
  ImageField: undefined,
  IntegerField: 0,
  IPAdapterModelField: undefined,
  LoRAModelField: undefined,
  MainModelField: undefined,
  SchedulerField: 'euler',
  SDXLMainModelField: undefined,
  SDXLRefinerModelField: undefined,
  StringField: '',
  T2IAdapterModelField: undefined,
  VAEModelField: undefined,
  ControlNetModelField: undefined,
};

export const buildFieldInputInstance = (id: string, template: FieldInputTemplate): FieldInputInstance => {
  const fieldInstance: FieldInputInstance = {
    id,
    name: template.name,
    type: template.type,
    label: '',
    fieldKind: 'input' as const,
    value: template.default ?? get(FIELD_VALUE_FALLBACK_MAP, template.type.name),
  };

  return fieldInstance;
};

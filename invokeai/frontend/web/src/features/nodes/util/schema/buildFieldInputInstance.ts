import { get } from 'es-toolkit/compat';
import type { FieldInputInstance, FieldInputTemplate, FieldValue, StatefulFieldType } from 'features/nodes/types/field';

const FIELD_VALUE_FALLBACK_MAP: Record<StatefulFieldType['name'], FieldValue> = {
  EnumField: '',
  BoardField: undefined,
  BooleanField: false,
  ColorField: { r: 0, g: 0, b: 0, a: 1 },
  FloatField: 0,
  ImageField: undefined,
  IntegerField: 0,
  ModelIdentifierField: undefined,
  SchedulerField: 'dpmpp_3m_k',
  StringField: '',
  FloatGeneratorField: undefined,
  IntegerGeneratorField: undefined,
  StringGeneratorField: undefined,
  ImageGeneratorField: undefined,
  QwenImageMainModelField: undefined,
  QwenImageVAEModelField: undefined,
  Qwen2_5VLModelField: undefined,
};

export const buildFieldInputInstance = (id: string, template: FieldInputTemplate): FieldInputInstance => {
  const fieldInstance: FieldInputInstance = {
    name: template.name,
    label: '',
    description: '',
    value: template.default ?? get(FIELD_VALUE_FALLBACK_MAP, template.type.name),
  };

  return fieldInstance;
};

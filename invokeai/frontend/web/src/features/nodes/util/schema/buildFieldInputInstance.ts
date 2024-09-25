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
  ModelIdentifierField: undefined,
  MainModelField: undefined,
  SchedulerField: 'dpmpp_3m_k',
  SDXLMainModelField: undefined,
  FluxMainModelField: undefined,
  SDXLRefinerModelField: undefined,
  StringField: '',
  T2IAdapterModelField: undefined,
  SpandrelImageToImageModelField: undefined,
  VAEModelField: undefined,
  ControlNetModelField: undefined,
  T5EncoderModelField: undefined,
  FluxVAEModelField: undefined,
  CLIPEmbedModelField: undefined,
};

export const buildFieldInputInstance = (id: string, template: FieldInputTemplate): FieldInputInstance => {
  const fieldInstance: FieldInputInstance = {
    name: template.name,
    label: '',
    value: template.default ?? get(FIELD_VALUE_FALLBACK_MAP, template.type.name),
  };

  return fieldInstance;
};

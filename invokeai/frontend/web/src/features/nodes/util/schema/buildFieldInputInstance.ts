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
  IPAdapterModelField: undefined,
  LoRAModelField: undefined,
  LLaVAModelField: undefined,
  ModelIdentifierField: undefined,
  MainModelField: undefined,
  SchedulerField: 'dpmpp_3m_k',
  SDXLMainModelField: undefined,
  FluxMainModelField: undefined,
  BriaMainModelField: undefined,
  BriaControlNetModelField: undefined,
  SD3MainModelField: undefined,
  CogView4MainModelField: undefined,
  SDXLRefinerModelField: undefined,
  StringField: '',
  T2IAdapterModelField: undefined,
  SpandrelImageToImageModelField: undefined,
  VAEModelField: undefined,
  ControlNetModelField: undefined,
  T5EncoderModelField: undefined,
  FluxVAEModelField: undefined,
  CLIPEmbedModelField: undefined,
  CLIPLEmbedModelField: undefined,
  CLIPGEmbedModelField: undefined,
  ControlLoRAModelField: undefined,
  SigLipModelField: undefined,
  FluxReduxModelField: undefined,
  Imagen3ModelField: undefined,
  Imagen4ModelField: undefined,
  ChatGPT4oModelField: undefined,
  FluxKontextModelField: undefined,
  FloatGeneratorField: undefined,
  IntegerGeneratorField: undefined,
  StringGeneratorField: undefined,
  ImageGeneratorField: undefined,
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

import { FieldParseError } from 'features/nodes/types/error';
import type {
  BoardFieldInputTemplate,
  BooleanFieldInputTemplate,
  CLIPEmbedModelFieldInputTemplate,
  CLIPGEmbedModelFieldInputTemplate,
  CLIPLEmbedModelFieldInputTemplate,
  ColorFieldInputTemplate,
  ControlLoRAModelFieldInputTemplate,
  ControlNetModelFieldInputTemplate,
  EnumFieldInputTemplate,
  FieldInputTemplate,
  FieldType,
  FloatFieldCollectionInputTemplate,
  FloatFieldInputTemplate,
  FloatGeneratorFieldInputTemplate,
  FluxMainModelFieldInputTemplate,
  FluxVAEModelFieldInputTemplate,
  ImageFieldCollectionInputTemplate,
  ImageFieldInputTemplate,
  IntegerFieldCollectionInputTemplate,
  IntegerFieldInputTemplate,
  IntegerGeneratorFieldInputTemplate,
  IPAdapterModelFieldInputTemplate,
  LoRAModelFieldInputTemplate,
  MainModelFieldInputTemplate,
  ModelIdentifierFieldInputTemplate,
  SchedulerFieldInputTemplate,
  SD3MainModelFieldInputTemplate,
  SDXLMainModelFieldInputTemplate,
  SDXLRefinerModelFieldInputTemplate,
  SpandrelImageToImageModelFieldInputTemplate,
  StatefulFieldType,
  StatelessFieldInputTemplate,
  StringFieldCollectionInputTemplate,
  StringFieldInputTemplate,
  StringGeneratorFieldInputTemplate,
  T2IAdapterModelFieldInputTemplate,
  T5EncoderModelFieldInputTemplate,
  VAEModelFieldInputTemplate,
} from 'features/nodes/types/field';
import {
  getFloatGeneratorArithmeticSequenceDefaults,
  getIntegerGeneratorArithmeticSequenceDefaults,
  getStringGeneratorParseStringDefaults,
  isFloatCollectionFieldType,
  isImageCollectionFieldType,
  isIntegerCollectionFieldType,
  isStatefulFieldType,
  isStringCollectionFieldType,
} from 'features/nodes/types/field';
import type { InvocationFieldSchema } from 'features/nodes/types/openapi';
import { isSchemaObject } from 'features/nodes/types/openapi';
import { t } from 'i18next';
import { isNumber, startCase } from 'lodash-es';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type FieldInputTemplateBuilder<T extends FieldInputTemplate = any> = // valid `any`!
  (arg: { schemaObject: InvocationFieldSchema; baseField: Omit<T, 'type' | 'default'>; fieldType: T['type'] }) => T;

const buildIntegerFieldInputTemplate: FieldInputTemplateBuilder<IntegerFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: IntegerFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? 0,
  };

  if (schemaObject.multipleOf !== undefined) {
    template.multipleOf = schemaObject.multipleOf;
  }

  if (schemaObject.maximum !== undefined) {
    template.maximum = schemaObject.maximum;
  }

  if (schemaObject.exclusiveMaximum !== undefined && isNumber(schemaObject.exclusiveMaximum)) {
    template.exclusiveMaximum = schemaObject.exclusiveMaximum;
  }

  if (schemaObject.minimum !== undefined) {
    template.minimum = schemaObject.minimum;
  }

  if (schemaObject.exclusiveMinimum !== undefined && isNumber(schemaObject.exclusiveMinimum)) {
    template.exclusiveMinimum = schemaObject.exclusiveMinimum;
  }

  return template;
};

const buildIntegerFieldCollectionInputTemplate: FieldInputTemplateBuilder<IntegerFieldCollectionInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: IntegerFieldCollectionInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? (schemaObject.orig_required ? [] : undefined),
  };

  if (schemaObject.minItems !== undefined) {
    template.minItems = schemaObject.minItems;
  }

  if (schemaObject.maxItems !== undefined) {
    template.maxItems = schemaObject.maxItems;
  }

  if (schemaObject.multipleOf !== undefined) {
    template.multipleOf = schemaObject.multipleOf;
  }

  if (schemaObject.maximum !== undefined) {
    template.maximum = schemaObject.maximum;
  }

  if (schemaObject.exclusiveMaximum !== undefined && isNumber(schemaObject.exclusiveMaximum)) {
    template.exclusiveMaximum = schemaObject.exclusiveMaximum;
  }

  if (schemaObject.minimum !== undefined) {
    template.minimum = schemaObject.minimum;
  }

  if (schemaObject.exclusiveMinimum !== undefined && isNumber(schemaObject.exclusiveMinimum)) {
    template.exclusiveMinimum = schemaObject.exclusiveMinimum;
  }

  return template;
};

const buildFloatFieldInputTemplate: FieldInputTemplateBuilder<FloatFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: FloatFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? 0,
  };

  if (schemaObject.multipleOf !== undefined) {
    template.multipleOf = schemaObject.multipleOf;
  }

  if (schemaObject.maximum !== undefined) {
    template.maximum = schemaObject.maximum;
  }

  if (schemaObject.exclusiveMaximum !== undefined && isNumber(schemaObject.exclusiveMaximum)) {
    template.exclusiveMaximum = schemaObject.exclusiveMaximum;
  }

  if (schemaObject.minimum !== undefined) {
    template.minimum = schemaObject.minimum;
  }

  if (schemaObject.exclusiveMinimum !== undefined && isNumber(schemaObject.exclusiveMinimum)) {
    template.exclusiveMinimum = schemaObject.exclusiveMinimum;
  }

  return template;
};

const buildFloatFieldCollectionInputTemplate: FieldInputTemplateBuilder<FloatFieldCollectionInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: FloatFieldCollectionInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? (schemaObject.orig_required ? [] : undefined),
  };

  if (schemaObject.minItems !== undefined) {
    template.minItems = schemaObject.minItems;
  }

  if (schemaObject.maxItems !== undefined) {
    template.maxItems = schemaObject.maxItems;
  }

  if (schemaObject.multipleOf !== undefined) {
    template.multipleOf = schemaObject.multipleOf;
  }

  if (schemaObject.maximum !== undefined) {
    template.maximum = schemaObject.maximum;
  }

  if (schemaObject.exclusiveMaximum !== undefined && isNumber(schemaObject.exclusiveMaximum)) {
    template.exclusiveMaximum = schemaObject.exclusiveMaximum;
  }

  if (schemaObject.minimum !== undefined) {
    template.minimum = schemaObject.minimum;
  }

  if (schemaObject.exclusiveMinimum !== undefined && isNumber(schemaObject.exclusiveMinimum)) {
    template.exclusiveMinimum = schemaObject.exclusiveMinimum;
  }

  return template;
};

const buildStringFieldInputTemplate: FieldInputTemplateBuilder<StringFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: StringFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? '',
  };

  if (schemaObject.minLength !== undefined) {
    template.minLength = schemaObject.minLength;
  }

  if (schemaObject.maxLength !== undefined) {
    template.maxLength = schemaObject.maxLength;
  }

  return template;
};

const buildStringFieldCollectionInputTemplate: FieldInputTemplateBuilder<StringFieldCollectionInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: StringFieldCollectionInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? (schemaObject.orig_required ? [] : undefined),
  };

  if (schemaObject.minLength !== undefined) {
    template.minLength = schemaObject.minLength;
  }

  if (schemaObject.maxLength !== undefined) {
    template.maxLength = schemaObject.maxLength;
  }

  if (schemaObject.minItems !== undefined) {
    template.minItems = schemaObject.minItems;
  }

  if (schemaObject.maxItems !== undefined) {
    template.maxItems = schemaObject.maxItems;
  }

  return template;
};

const buildBooleanFieldInputTemplate: FieldInputTemplateBuilder<BooleanFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: BooleanFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? false,
  };

  return template;
};

const buildModelIdentifierFieldInputTemplate: FieldInputTemplateBuilder<ModelIdentifierFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: ModelIdentifierFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildMainModelFieldInputTemplate: FieldInputTemplateBuilder<MainModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: MainModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildSDXLMainModelFieldInputTemplate: FieldInputTemplateBuilder<SDXLMainModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: SDXLMainModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildFluxMainModelFieldInputTemplate: FieldInputTemplateBuilder<FluxMainModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: FluxMainModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildSD3MainModelFieldInputTemplate: FieldInputTemplateBuilder<SD3MainModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: SD3MainModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildRefinerModelFieldInputTemplate: FieldInputTemplateBuilder<SDXLRefinerModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: SDXLRefinerModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildVAEModelFieldInputTemplate: FieldInputTemplateBuilder<VAEModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: VAEModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildT5EncoderModelFieldInputTemplate: FieldInputTemplateBuilder<T5EncoderModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: T5EncoderModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildCLIPEmbedModelFieldInputTemplate: FieldInputTemplateBuilder<CLIPEmbedModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: CLIPEmbedModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildCLIPLEmbedModelFieldInputTemplate: FieldInputTemplateBuilder<CLIPLEmbedModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: CLIPLEmbedModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildCLIPGEmbedModelFieldInputTemplate: FieldInputTemplateBuilder<CLIPGEmbedModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: CLIPGEmbedModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildControlLoRAModelFieldInputTemplate: FieldInputTemplateBuilder<ControlLoRAModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: ControlLoRAModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildFluxVAEModelFieldInputTemplate: FieldInputTemplateBuilder<FluxVAEModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: FluxVAEModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildLoRAModelFieldInputTemplate: FieldInputTemplateBuilder<LoRAModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: LoRAModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildControlNetModelFieldInputTemplate: FieldInputTemplateBuilder<ControlNetModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: ControlNetModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildIPAdapterModelFieldInputTemplate: FieldInputTemplateBuilder<IPAdapterModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: IPAdapterModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildT2IAdapterModelFieldInputTemplate: FieldInputTemplateBuilder<T2IAdapterModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: T2IAdapterModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildSpandrelImageToImageModelFieldInputTemplate: FieldInputTemplateBuilder<
  SpandrelImageToImageModelFieldInputTemplate
> = ({ schemaObject, baseField, fieldType }) => {
  const template: SpandrelImageToImageModelFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};
const buildBoardFieldInputTemplate: FieldInputTemplateBuilder<BoardFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: BoardFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildImageFieldInputTemplate: FieldInputTemplateBuilder<ImageFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: ImageFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildImageFieldCollectionInputTemplate: FieldInputTemplateBuilder<ImageFieldCollectionInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: ImageFieldCollectionInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? (schemaObject.orig_required ? [] : undefined),
  };

  if (schemaObject.minItems !== undefined) {
    template.minItems = schemaObject.minItems;
  }

  if (schemaObject.maxItems !== undefined) {
    template.maxItems = schemaObject.maxItems;
  }

  return template;
};

const buildEnumFieldInputTemplate: FieldInputTemplateBuilder<EnumFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  let options: EnumFieldInputTemplate['options'] = [];
  if (schemaObject.anyOf) {
    const filteredAnyOf = schemaObject.anyOf.filter((i) => {
      if (isSchemaObject(i)) {
        if (i.type === 'null') {
          return false;
        }
      }
      return true;
    });
    const firstAnyOf = filteredAnyOf[0];
    if (filteredAnyOf.length !== 1 || !isSchemaObject(firstAnyOf)) {
      options = [];
    } else {
      options = firstAnyOf.enum ?? [];
    }
  } else if (schemaObject.const) {
    options = [schemaObject.const];
  } else {
    options = schemaObject.enum ?? [];
  }
  if (options.length === 0) {
    throw new FieldParseError(t('nodes.unableToExtractEnumOptions'));
  }
  const template: EnumFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    options,
    ui_choice_labels: schemaObject.ui_choice_labels,
    default: schemaObject.default ?? options[0],
  };

  return template;
};

const buildColorFieldInputTemplate: FieldInputTemplateBuilder<ColorFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: ColorFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? { r: 127, g: 127, b: 127, a: 255 },
  };

  return template;
};

const buildSchedulerFieldInputTemplate: FieldInputTemplateBuilder<SchedulerFieldInputTemplate> = ({
  schemaObject,
  baseField,
  fieldType,
}) => {
  const template: SchedulerFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: schemaObject.default ?? 'dpmpp_3m_k',
  };

  return template;
};

const buildFloatGeneratorFieldInputTemplate: FieldInputTemplateBuilder<FloatGeneratorFieldInputTemplate> = ({
  // schemaObject,
  baseField,
  fieldType,
}) => {
  const template: FloatGeneratorFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: getFloatGeneratorArithmeticSequenceDefaults(),
  };

  return template;
};

const buildIntegerGeneratorFieldInputTemplate: FieldInputTemplateBuilder<IntegerGeneratorFieldInputTemplate> = ({
  // schemaObject,
  baseField,
  fieldType,
}) => {
  const template: IntegerGeneratorFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: getIntegerGeneratorArithmeticSequenceDefaults(),
  };

  return template;
};

const buildStringGeneratorFieldInputTemplate: FieldInputTemplateBuilder<StringGeneratorFieldInputTemplate> = ({
  // schemaObject,
  baseField,
  fieldType,
}) => {
  const template: StringGeneratorFieldInputTemplate = {
    ...baseField,
    type: fieldType,
    default: getStringGeneratorParseStringDefaults(),
  };

  return template;
};

export const TEMPLATE_BUILDER_MAP: Record<StatefulFieldType['name'], FieldInputTemplateBuilder> = {
  BoardField: buildBoardFieldInputTemplate,
  BooleanField: buildBooleanFieldInputTemplate,
  ColorField: buildColorFieldInputTemplate,
  ControlNetModelField: buildControlNetModelFieldInputTemplate,
  EnumField: buildEnumFieldInputTemplate,
  FloatField: buildFloatFieldInputTemplate,
  ImageField: buildImageFieldInputTemplate,
  IntegerField: buildIntegerFieldInputTemplate,
  IPAdapterModelField: buildIPAdapterModelFieldInputTemplate,
  LoRAModelField: buildLoRAModelFieldInputTemplate,
  ModelIdentifierField: buildModelIdentifierFieldInputTemplate,
  MainModelField: buildMainModelFieldInputTemplate,
  SchedulerField: buildSchedulerFieldInputTemplate,
  SDXLMainModelField: buildSDXLMainModelFieldInputTemplate,
  SD3MainModelField: buildSD3MainModelFieldInputTemplate,
  FluxMainModelField: buildFluxMainModelFieldInputTemplate,
  SDXLRefinerModelField: buildRefinerModelFieldInputTemplate,
  StringField: buildStringFieldInputTemplate,
  T2IAdapterModelField: buildT2IAdapterModelFieldInputTemplate,
  SpandrelImageToImageModelField: buildSpandrelImageToImageModelFieldInputTemplate,
  VAEModelField: buildVAEModelFieldInputTemplate,
  T5EncoderModelField: buildT5EncoderModelFieldInputTemplate,
  CLIPEmbedModelField: buildCLIPEmbedModelFieldInputTemplate,
  CLIPLEmbedModelField: buildCLIPLEmbedModelFieldInputTemplate,
  CLIPGEmbedModelField: buildCLIPGEmbedModelFieldInputTemplate,
  FluxVAEModelField: buildFluxVAEModelFieldInputTemplate,
  ControlLoRAModelField: buildControlLoRAModelFieldInputTemplate,
  FloatGeneratorField: buildFloatGeneratorFieldInputTemplate,
  IntegerGeneratorField: buildIntegerGeneratorFieldInputTemplate,
  StringGeneratorField: buildStringGeneratorFieldInputTemplate,
} as const;

export const buildFieldInputTemplate = (
  fieldSchema: InvocationFieldSchema,
  fieldName: string,
  fieldType: FieldType
): FieldInputTemplate => {
  const { input, ui_hidden, ui_component, ui_type, ui_order, ui_choice_labels, orig_required: required } = fieldSchema;

  // This is the base field template that is common to all fields. The builder function will add all other
  // properties to this template.
  const baseField: Omit<FieldInputTemplate, 'type' | 'default'> = {
    name: fieldName,
    title: fieldSchema.title ?? (fieldName ? startCase(fieldName) : ''),
    required,
    description: fieldSchema.description ?? '',
    fieldKind: 'input' as const,
    input,
    ui_hidden,
    ui_component,
    ui_type,
    ui_order,
    ui_choice_labels,
  };

  if (isStatefulFieldType(fieldType)) {
    if (isImageCollectionFieldType(fieldType)) {
      return buildImageFieldCollectionInputTemplate({
        schemaObject: fieldSchema,
        baseField,
        fieldType,
      });
    } else if (isStringCollectionFieldType(fieldType)) {
      return buildStringFieldCollectionInputTemplate({
        schemaObject: fieldSchema,
        baseField,
        fieldType,
      });
    } else if (isIntegerCollectionFieldType(fieldType)) {
      return buildIntegerFieldCollectionInputTemplate({
        schemaObject: fieldSchema,
        baseField,
        fieldType,
      });
    } else if (isFloatCollectionFieldType(fieldType)) {
      return buildFloatFieldCollectionInputTemplate({
        schemaObject: fieldSchema,
        baseField,
        fieldType,
      });
    } else {
      const builder = TEMPLATE_BUILDER_MAP[fieldType.name];
      const template = builder({
        schemaObject: fieldSchema,
        baseField,
        fieldType,
      });

      return template;
    }
  } else {
    // This is a StatelessField, create it directly.
    const template: StatelessFieldInputTemplate = {
      ...baseField,
      input: 'connection', // stateless --> connection only inputs
      type: fieldType,
      default: undefined, // stateless --> no default value
    };

    return template;
  }
};

import { FieldParseError } from 'features/nodes/types/error';
import type {
  BoardFieldInputTemplate,
  BooleanFieldInputTemplate,
  ColorFieldInputTemplate,
  ControlNetModelFieldInputTemplate,
  EnumFieldInputTemplate,
  FieldInputTemplate,
  FieldType,
  FloatFieldInputTemplate,
  ImageFieldInputTemplate,
  IntegerFieldInputTemplate,
  IPAdapterModelFieldInputTemplate,
  LoRAModelFieldInputTemplate,
  MainModelFieldInputTemplate,
  SchedulerFieldInputTemplate,
  SDXLMainModelFieldInputTemplate,
  SDXLRefinerModelFieldInputTemplate,
  StatefulFieldType,
  StatelessFieldInputTemplate,
  StringFieldInputTemplate,
  T2IAdapterModelFieldInputTemplate,
  VAEModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { isStatefulFieldType } from 'features/nodes/types/field';
import type { InvocationFieldSchema } from 'features/nodes/types/openapi';
import { isSchemaObject } from 'features/nodes/types/openapi';
import { t } from 'i18next';
import { isNumber, startCase } from 'lodash-es';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type FieldInputTemplateBuilder<T extends FieldInputTemplate = any> = // valid `any`!
  (arg: {
    schemaObject: InvocationFieldSchema;
    baseField: Omit<T, 'type'>;
    isCollection: boolean;
    isCollectionOrScalar: boolean;
  }) => T;

const buildIntegerFieldInputTemplate: FieldInputTemplateBuilder<IntegerFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: IntegerFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'IntegerField',
      isCollection,
      isCollectionOrScalar,
    },
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

const buildFloatFieldInputTemplate: FieldInputTemplateBuilder<FloatFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: FloatFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'FloatField',
      isCollection,
      isCollectionOrScalar,
    },
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

const buildStringFieldInputTemplate: FieldInputTemplateBuilder<StringFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: StringFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'StringField',
      isCollection,
      isCollectionOrScalar,
    },
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

const buildBooleanFieldInputTemplate: FieldInputTemplateBuilder<BooleanFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: BooleanFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'BooleanField',
      isCollection,
      isCollectionOrScalar,
    },
    default: schemaObject.default ?? false,
  };

  return template;
};

const buildMainModelFieldInputTemplate: FieldInputTemplateBuilder<MainModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: MainModelFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'MainModelField',
      isCollection,
      isCollectionOrScalar,
    },
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildSDXLMainModelFieldInputTemplate: FieldInputTemplateBuilder<SDXLMainModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: SDXLMainModelFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'SDXLMainModelField',
      isCollection,
      isCollectionOrScalar,
    },
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildRefinerModelFieldInputTemplate: FieldInputTemplateBuilder<SDXLRefinerModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: SDXLRefinerModelFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'SDXLRefinerModelField',
      isCollection,
      isCollectionOrScalar,
    },
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildVAEModelFieldInputTemplate: FieldInputTemplateBuilder<VAEModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: VAEModelFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'VAEModelField',
      isCollection,
      isCollectionOrScalar,
    },
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildLoRAModelFieldInputTemplate: FieldInputTemplateBuilder<LoRAModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: LoRAModelFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'LoRAModelField',
      isCollection,
      isCollectionOrScalar,
    },
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildControlNetModelFieldInputTemplate: FieldInputTemplateBuilder<ControlNetModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: ControlNetModelFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'ControlNetModelField',
      isCollection,
      isCollectionOrScalar,
    },
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildIPAdapterModelFieldInputTemplate: FieldInputTemplateBuilder<IPAdapterModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: IPAdapterModelFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'IPAdapterModelField',
      isCollection,
      isCollectionOrScalar,
    },
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildT2IAdapterModelFieldInputTemplate: FieldInputTemplateBuilder<T2IAdapterModelFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: T2IAdapterModelFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'T2IAdapterModelField',
      isCollection,
      isCollectionOrScalar,
    },
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildBoardFieldInputTemplate: FieldInputTemplateBuilder<BoardFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: BoardFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'BoardField',
      isCollection,
      isCollectionOrScalar,
    },
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildImageFieldInputTemplate: FieldInputTemplateBuilder<ImageFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: ImageFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'ImageField',
      isCollection,
      isCollectionOrScalar,
    },
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildEnumFieldInputTemplate: FieldInputTemplateBuilder<EnumFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
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
    type: {
      name: 'EnumField',
      isCollection,
      isCollectionOrScalar,
    },
    options,
    ui_choice_labels: schemaObject.ui_choice_labels,
    default: schemaObject.default ?? options[0],
  };

  return template;
};

const buildColorFieldInputTemplate: FieldInputTemplateBuilder<ColorFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: ColorFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'ColorField',
      isCollection,
      isCollectionOrScalar,
    },
    default: schemaObject.default ?? { r: 127, g: 127, b: 127, a: 255 },
  };

  return template;
};

const buildSchedulerFieldInputTemplate: FieldInputTemplateBuilder<SchedulerFieldInputTemplate> = ({
  schemaObject,
  baseField,
  isCollection,
  isCollectionOrScalar,
}) => {
  const template: SchedulerFieldInputTemplate = {
    ...baseField,
    type: {
      name: 'SchedulerField',
      isCollection,
      isCollectionOrScalar,
    },
    default: schemaObject.default ?? 'euler',
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
  MainModelField: buildMainModelFieldInputTemplate,
  SchedulerField: buildSchedulerFieldInputTemplate,
  SDXLMainModelField: buildSDXLMainModelFieldInputTemplate,
  SDXLRefinerModelField: buildRefinerModelFieldInputTemplate,
  StringField: buildStringFieldInputTemplate,
  T2IAdapterModelField: buildT2IAdapterModelFieldInputTemplate,
  VAEModelField: buildVAEModelFieldInputTemplate,
};

export const buildFieldInputTemplate = (
  fieldSchema: InvocationFieldSchema,
  fieldName: string,
  fieldType: FieldType
): FieldInputTemplate => {
  const { input, ui_hidden, ui_component, ui_type, ui_order, ui_choice_labels, orig_required: required } = fieldSchema;

  // This is the base field template that is common to all fields. The builder function will add all other
  // properties to this template.
  const baseField: Omit<FieldInputTemplate, 'type'> = {
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
    const builder = TEMPLATE_BUILDER_MAP[fieldType.name];
    return builder({
      schemaObject: fieldSchema,
      baseField,
      isCollection: fieldType.isCollection,
      isCollectionOrScalar: fieldType.isCollectionOrScalar,
    });
  }

  // This is a StatelessField, create it directly.
  const template: StatelessFieldInputTemplate = {
    ...baseField,
    input: 'connection', // stateless --> connection only inputs
    type: fieldType,
    default: undefined, // stateless --> no default value
  };
  return template;
};

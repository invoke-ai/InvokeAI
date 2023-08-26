import { InputFieldTemplate, InputFieldValue } from '../types/types';

export const buildInputFieldValue = (
  id: string,
  template: InputFieldTemplate
): InputFieldValue => {
  const fieldValue: InputFieldValue = {
    id,
    name: template.name,
    type: template.type,
    label: '',
    fieldKind: 'input',
  };

  if (template.type === 'string') {
    fieldValue.value = template.default ?? '';
  }

  if (template.type === 'integer') {
    fieldValue.value = template.default ?? 0;
  }

  if (template.type === 'float') {
    fieldValue.value = template.default ?? 0;
  }

  if (template.type === 'boolean') {
    fieldValue.value = template.default ?? false;
  }

  if (template.type === 'enum') {
    if (template.enumType === 'number') {
      fieldValue.value = template.default ?? 0;
    }
    if (template.enumType === 'string') {
      fieldValue.value = template.default ?? '';
    }
  }

  if (template.type === 'Collection') {
    fieldValue.value = template.default ?? 1;
  }

  if (template.type === 'ImageField') {
    fieldValue.value = undefined;
  }

  if (template.type === 'ImageCollection') {
    fieldValue.value = [];
  }

  if (template.type === 'DenoiseMaskField') {
    fieldValue.value = undefined;
  }

  if (template.type === 'LatentsField') {
    fieldValue.value = undefined;
  }

  if (template.type === 'ConditioningField') {
    fieldValue.value = undefined;
  }

  if (template.type === 'UNetField') {
    fieldValue.value = undefined;
  }

  if (template.type === 'ClipField') {
    fieldValue.value = undefined;
  }

  if (template.type === 'VaeField') {
    fieldValue.value = undefined;
  }

  if (template.type === 'ControlField') {
    fieldValue.value = undefined;
  }

  if (template.type === 'MainModelField') {
    fieldValue.value = undefined;
  }

  if (template.type === 'SDXLRefinerModelField') {
    fieldValue.value = undefined;
  }

  if (template.type === 'VaeModelField') {
    fieldValue.value = undefined;
  }

  if (template.type === 'LoRAModelField') {
    fieldValue.value = undefined;
  }

  if (template.type === 'ControlNetModelField') {
    fieldValue.value = undefined;
  }

  if (template.type === 'Scheduler') {
    fieldValue.value = 'euler';
  }

  return fieldValue;
};

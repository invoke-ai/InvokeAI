import { InputFieldTemplate, InputFieldValue } from '../types/types';

export const buildInputFieldValue = (
  id: string,
  template: InputFieldTemplate
): InputFieldValue => {
  const fieldValue: InputFieldValue = {
    id,
    name: template.name,
    type: template.type,
  };

  if (template.inputRequirement !== 'never') {
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

    if (template.type === 'array') {
      fieldValue.value = template.default ?? 1;
    }

    if (template.type === 'image') {
      fieldValue.value = undefined;
    }

    if (template.type === 'image_collection') {
      fieldValue.value = [];
    }

    if (template.type === 'latents') {
      fieldValue.value = undefined;
    }

    if (template.type === 'conditioning') {
      fieldValue.value = undefined;
    }

    if (template.type === 'unet') {
      fieldValue.value = undefined;
    }

    if (template.type === 'clip') {
      fieldValue.value = undefined;
    }

    if (template.type === 'vae') {
      fieldValue.value = undefined;
    }

    if (template.type === 'control') {
      fieldValue.value = undefined;
    }

    if (template.type === 'model') {
      fieldValue.value = undefined;
    }

    if (template.type === 'vae_model') {
      fieldValue.value = undefined;
    }

    if (template.type === 'lora_model') {
      fieldValue.value = undefined;
    }

    if (template.type === 'controlnet_model') {
      fieldValue.value = undefined;
    }
  }

  return fieldValue;
};

import { BaseModelType, VAEModelField } from 'services/api/types';

/**
 * Crudely converts a model id to a main model field
 * TODO: Make better
 */
export const modelIdToVAEModelField = (modelId: string): VAEModelField => {
  const [base_model, model_type, model_name] = modelId.split('/');

  const field: VAEModelField = {
    base_model: base_model as BaseModelType,
    model_name,
  };

  return field;
};

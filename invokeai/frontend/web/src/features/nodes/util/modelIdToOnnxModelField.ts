import { BaseModelType, OnnxModelField, ModelType } from 'services/api/types';

/**
 * Crudely converts a model id to a main model field
 * TODO: Make better
 */
export const modelIdToOnnxModelField = (modelId: string): OnnxModelField => {
  const [base_model, model_type, model_name] = modelId.split('/');

  const field: OnnxModelField = {
    base_model: base_model as BaseModelType,
    model_name,
    model_type: model_type as ModelType,
  };

  return field;
};

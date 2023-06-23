import { BaseModelType, PipelineModelField } from 'services/api/types';

/**
 * Crudely converts a model id to a pipeline model field
 * TODO: Make better
 */
export const modelIdToPipelineModelField = (
  modelId: string
): PipelineModelField => {
  const [base_model, model_type, model_name] = modelId.split('/');

  const field: PipelineModelField = {
    base_model: base_model as BaseModelType,
    model_name,
  };

  return field;
};

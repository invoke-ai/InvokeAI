import { BaseModelType, ControlNetModelField } from 'services/api/types';

export const modelIdToControlNetModelField = (
  controlNetModelId: string
): ControlNetModelField => {
  const [base_model, model_type, model_name] = controlNetModelId.split('/');

  const field: ControlNetModelField = {
    base_model: base_model as BaseModelType,
    model_name,
  };

  return field;
};

import { BaseModelType, LoRAModelField } from 'services/api/types';

export const modelIdToLoRAModelField = (loraId: string): LoRAModelField => {
  const [base_model, model_type, model_name] = loraId.split('/');

  const field: LoRAModelField = {
    base_model: base_model as BaseModelType,
    model_name,
  };

  return field;
};

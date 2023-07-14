import { LoRAModelParam, zLoRAModel } from '../types/parameterSchemas';

export const modelIdToLoRAModelParam = (
  loraId: string
): LoRAModelParam | undefined => {
  const [base_model, model_type, model_name] = loraId.split('/');

  const result = zLoRAModel.safeParse({
    base_model,
    model_name,
  });

  if (!result.success) {
    return;
  }

  return result.data;
};

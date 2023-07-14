import { VaeModelParam, zVaeModel } from '../types/parameterSchemas';

export const modelIdToVAEModelParam = (
  modelId: string
): VaeModelParam | undefined => {
  const [base_model, model_type, model_name] = modelId.split('/');

  const result = zVaeModel.safeParse({
    base_model,
    model_name,
  });

  if (!result.success) {
    return;
  }

  return result.data;
};

import {
  MainModelParam,
  zMainModel,
} from 'features/parameters/types/parameterSchemas';

export const modelIdToMainModelParam = (
  modelId: string
): MainModelParam | undefined => {
  const [base_model, model_type, model_name] = modelId.split('/');

  const result = zMainModel.safeParse({
    base_model,
    model_name,
  });

  if (!result.success) {
    return;
  }

  return result.data;
};

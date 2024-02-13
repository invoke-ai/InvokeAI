import { logger } from 'app/logging/logger';
import type { ParameterVAEModel } from 'features/parameters/types/parameterSchemas';
import { zParameterVAEModel } from 'features/parameters/types/parameterSchemas';

export const modelIdToVAEModelParam = (vaeModelId: string): ParameterVAEModel | undefined => {
  const log = logger('models');
  const [base_model, _model_type, model_name] = vaeModelId.split('/');

  const result = zParameterVAEModel.safeParse({
    base_model,
    model_name,
  });

  if (!result.success) {
    log.error(
      {
        vaeModelId,
        errors: result.error.format(),
      },
      'Failed to parse VAE model id'
    );
    return;
  }

  return result.data;
};

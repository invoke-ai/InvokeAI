import { logger } from 'app/logging/logger';
import { VaeModelParam, zVaeModel } from '../types/parameterSchemas';

export const modelIdToVAEModelParam = (
  vaeModelId: string
): VaeModelParam | undefined => {
  const log = logger('models');
  const [base_model, _model_type, model_name] = vaeModelId.split('/');

  const result = zVaeModel.safeParse({
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

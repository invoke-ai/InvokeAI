import { VaeModelParam, zVaeModel } from '../types/parameterSchemas';
import { log } from 'app/logging/useLogger';

const moduleLog = log.child({ module: 'models' });

export const modelIdToVAEModelParam = (
  vaeModelId: string
): VaeModelParam | undefined => {
  const [base_model, model_type, model_name] = vaeModelId.split('/');

  const result = zVaeModel.safeParse({
    base_model,
    model_name,
  });

  if (!result.success) {
    moduleLog.error(
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

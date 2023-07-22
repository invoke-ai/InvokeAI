import { logger } from 'app/logging/logger';
import { LoRAModelParam, zLoRAModel } from '../types/parameterSchemas';

export const modelIdToLoRAModelParam = (
  loraModelId: string
): LoRAModelParam | undefined => {
  const log = logger('models');

  const [base_model, _model_type, model_name] = loraModelId.split('/');

  const result = zLoRAModel.safeParse({
    base_model,
    model_name,
  });

  if (!result.success) {
    log.error(
      {
        loraModelId,
        errors: result.error.format(),
      },
      'Failed to parse LoRA model id'
    );
    return;
  }

  return result.data;
};

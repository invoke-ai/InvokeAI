import { logger } from 'app/logging/logger';
import type { ParameterLoRAModel } from 'features/parameters/types/parameterSchemas';
import { zParameterLoRAModel } from 'features/parameters/types/parameterSchemas';

export const modelIdToLoRAModelParam = (loraModelId: string): ParameterLoRAModel | undefined => {
  const log = logger('models');

  const [base_model, _model_type, model_name] = loraModelId.split('/');

  const result = zParameterLoRAModel.safeParse({
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

import { logger } from 'app/logging/logger';
import type { ParameterModel } from 'features/parameters/types/parameterSchemas';
import { zParameterModel } from 'features/parameters/types/parameterSchemas';

export const modelIdToMainModelParam = (mainModelId: string): ParameterModel | undefined => {
  const log = logger('models');
  const [base_model, model_type, model_name] = mainModelId.split('/');

  const result = zParameterModel.safeParse({
    base_model,
    model_name,
    model_type,
  });

  if (!result.success) {
    log.error(
      {
        mainModelId,
        errors: result.error.format(),
      },
      'Failed to parse main model id'
    );
    return;
  }

  return result.data;
};

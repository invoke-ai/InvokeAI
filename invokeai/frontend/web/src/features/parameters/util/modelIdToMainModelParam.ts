import { logger } from 'app/logging/logger';
import {
  MainModelParam,
  zMainModel,
} from 'features/parameters/types/parameterSchemas';

export const modelIdToMainModelParam = (
  mainModelId: string
): MainModelParam | undefined => {
  const log = logger('models');
  const [base_model, _model_type, model_name] = mainModelId.split('/');

  const result = zMainModel.safeParse({
    base_model,
    model_name,
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

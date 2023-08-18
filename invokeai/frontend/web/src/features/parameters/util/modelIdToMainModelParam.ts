import { logger } from 'app/logging/logger';
import {
  MainModelParam,
  OnnxModelParam,
  zMainOrOnnxModel,
} from 'features/parameters/types/parameterSchemas';

export const modelIdToMainModelParam = (
  mainModelId: string
): OnnxModelParam | MainModelParam | undefined => {
  const log = logger('models');
  const [base_model, model_type, model_name] = mainModelId.split('/');

  const result = zMainOrOnnxModel.safeParse({
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

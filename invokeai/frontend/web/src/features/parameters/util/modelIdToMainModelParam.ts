import {
  MainModelParam,
  zMainModel,
} from 'features/parameters/types/parameterSchemas';
import { log } from 'app/logging/useLogger';

const moduleLog = log.child({ module: 'models' });

export const modelIdToMainModelParam = (
  mainModelId: string
): MainModelParam | undefined => {
  const [base_model, model_type, model_name] = mainModelId.split('/');

  const result = zMainModel.safeParse({
    base_model,
    model_name,
    model_type,
  });

  if (!result.success) {
    moduleLog.error(
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

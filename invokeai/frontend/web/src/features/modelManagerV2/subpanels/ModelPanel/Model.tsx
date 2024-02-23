import { useAppSelector } from 'app/store/storeHooks';

import { ModelEdit } from './ModelEdit';
import { ModelView } from './ModelView';

export const Model = () => {
  const selectedModelMode = useAppSelector((s) => s.modelmanagerV2.selectedModelMode);
  return selectedModelMode === 'view' ? <ModelView /> : <ModelEdit />;
};

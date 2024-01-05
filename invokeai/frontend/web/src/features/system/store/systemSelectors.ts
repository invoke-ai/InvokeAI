import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { selectSystemSlice } from 'features/system/store/systemSlice';

export const languageSelector = createMemoizedSelector(
  selectSystemSlice,
  (system) => system.language
);

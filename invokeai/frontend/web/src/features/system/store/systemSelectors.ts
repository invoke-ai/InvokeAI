import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';

export const languageSelector = createMemoizedSelector(
  stateSelector,
  ({ system }) => system.language
);

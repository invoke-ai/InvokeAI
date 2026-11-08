import { describe } from 'vitest';

import { createMemoryProjectDraftStore } from './draftStore';
import { testUnloadJournalScenarios } from './unloadJournal.scenarios';

describe('unload journal journeys with the memory draft store', () => {
  testUnloadJournalScenarios(() => Promise.resolve(createMemoryProjectDraftStore()));
});

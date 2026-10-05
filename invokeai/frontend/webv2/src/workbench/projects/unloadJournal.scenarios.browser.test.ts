import { captureAccountScope } from '@platform/state/accountLifecycle';
import { describe } from 'vitest';

import { createAccountOwnedProjectDraftStore } from './indexedDbDraftStore';
import { testUnloadJournalScenarios } from './unloadJournal.scenarios';

describe('unload journal journeys with IndexedDB', () => {
  testUnloadJournalScenarios(() => createAccountOwnedProjectDraftStore(captureAccountScope()));
});

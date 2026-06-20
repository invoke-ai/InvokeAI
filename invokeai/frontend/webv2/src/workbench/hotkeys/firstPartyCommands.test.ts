import { describe, expect, it } from 'vitest';

import { firstPartyHotkeyCatalog } from './catalog';
import {
  FIRST_PARTY_APP_COMMAND_IDS,
  FIRST_PARTY_COMMAND_IDS,
  FIRST_PARTY_IMAGE_RECALL_COMMAND_IDS,
} from './firstPartyCommands';

const centrallyOwnedCommandIds = new Set(FIRST_PARTY_COMMAND_IDS);

describe('first-party hotkey commands', () => {
  it('registers handlers for implemented app and shared image recall commands', () => {
    const sharedImageRecallCommands = new Set([
      'gallery.remix',
      'viewer.recallAll',
      'viewer.recallPrompts',
      'viewer.recallSeed',
      'viewer.remix',
      'viewer.useSize',
    ]);
    const expectedCentralCommands = new Set(
      firstPartyHotkeyCatalog
        .filter((hotkey) => hotkey.implemented !== false)
        .filter((hotkey) => hotkey.category === 'app' || sharedImageRecallCommands.has(hotkey.commandId))
        .map((hotkey) => hotkey.commandId)
    );

    expect(centrallyOwnedCommandIds).toEqual(expectedCentralCommands);
    expect(new Set(FIRST_PARTY_COMMAND_IDS).size).toBe(FIRST_PARTY_COMMAND_IDS.length);
    expect(new Set(FIRST_PARTY_APP_COMMAND_IDS).size).toBe(FIRST_PARTY_APP_COMMAND_IDS.length);
    expect(new Set(FIRST_PARTY_IMAGE_RECALL_COMMAND_IDS).size).toBe(FIRST_PARTY_IMAGE_RECALL_COMMAND_IDS.length);
  });
});

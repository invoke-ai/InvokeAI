import { firstPartyHotkeyCatalog } from '@workbench/hotkeys/catalog';
import { describe, expect, it } from 'vitest';

import { WORKFLOW_HOTKEYS } from './workflowHotkeys';

describe('WORKFLOW_HOTKEYS', () => {
  // Hotkey settings and the context menu's hints read the catalog; the editor registers its own list.
  it('registers every catalog workflow command with the catalog default keys', () => {
    const catalogDefaults = Object.fromEntries(
      firstPartyHotkeyCatalog
        .filter((hotkey) => hotkey.category === 'workflows')
        .map((hotkey) => [hotkey.id, hotkey.defaultKeys])
    );
    const editorDefaults = Object.fromEntries(WORKFLOW_HOTKEYS.map(({ defaultKeys, id }) => [id, [...defaultKeys]]));

    expect(editorDefaults).toEqual(catalogDefaults);
  });
});

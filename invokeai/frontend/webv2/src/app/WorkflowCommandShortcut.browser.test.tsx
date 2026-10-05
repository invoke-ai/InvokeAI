import type * as settingsStoreModule from '@workbench/settings/store';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { IS_MAC_OS } from '@workbench/hotkeys/keys';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { WorkflowCommandShortcut } from './WorkflowCommandShortcut';

const testPreferences = vi.hoisted(() => ({ customHotkeys: {} as Record<string, string[]> }));

vi.mock('@workbench/settings/store', async (importOriginal) => ({
  ...(await importOriginal<typeof settingsStoreModule>()),
  useWorkbenchPreferenceSelector: <Selected,>(selector: (preferences: typeof testPreferences) => Selected): Selected =>
    selector(testPreferences),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  testPreferences.customHotkeys = {};
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

/** The keycaps as read aloud: a glyph key reads as its spoken name. */
const keycaps = async (commandId: string) => {
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <WorkflowCommandShortcut commandId={commandId} />
      </ChakraProvider>
    )
  );

  return [...host.querySelectorAll('kbd')].map((kbd) => kbd.textContent);
};

describe('WorkflowCommandShortcut', () => {
  it("shows a workflow command's default binding in the platform's modifier names", async () => {
    expect(await keycaps('workflows.copySelection')).toEqual(IS_MAC_OS ? ['Command', 'c'] : ['ctrl', 'c']);
    expect(await keycaps('workflows.deleteSelection')).toEqual(['delete']);
  });

  it('follows a remapped binding and shows nothing for an unbound command', async () => {
    testPreferences.customHotkeys = { 'workflows.copySelection': [], 'workflows.pasteSelection': ['mod+shift+b'] };

    expect(await keycaps('workflows.pasteSelection')).toEqual(
      IS_MAC_OS ? ['Command', 'Shift', 'b'] : ['ctrl', 'shift', 'b']
    );
    expect(await keycaps('workflows.copySelection')).toEqual([]);
  });
});

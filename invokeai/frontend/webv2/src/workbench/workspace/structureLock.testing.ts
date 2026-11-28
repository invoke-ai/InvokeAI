import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { afterEach, beforeEach } from 'vitest';

/**
 * Runs each test of the calling scope with "Rearrange without Customize" on, where the shell's structural chrome
 * (menus, drag handles, float, dock and remove) works in normal use, as for an account that kept rearranging.
 */
export const rearrangeWithoutCustomizeInEachTest = (): void => {
  beforeEach(() => {
    // Preferences belong to an account; a harness without one gets a test account before its first write.
    if (captureAccountScope().accountId === null) {
      accountLifecycle.activate('rearranging-test-account');
    }
    // The write to the server fails in a test harness; the preference applies at once all the same.
    void patchWorkbenchPreferences({ rearrangeWithoutCustomize: true });
  });
  afterEach(() => {
    if (captureAccountScope().accountId !== null) {
      void patchWorkbenchPreferences({ rearrangeWithoutCustomize: false });
    }
  });
};

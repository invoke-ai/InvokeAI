import { beforeEach } from 'vitest';
import { server } from 'vitest/browser';

declare module 'vitest/browser' {
  interface BrowserCommands {
    resetBrowserPointer: () => Promise<void>;
  }
}

beforeEach(async () => {
  await server.commands.resetBrowserPointer();
});

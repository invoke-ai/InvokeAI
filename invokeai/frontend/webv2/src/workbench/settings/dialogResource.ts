import { createDeferredResource } from '@workbench/deferredResource';

export const settingsDialogResource = createDeferredResource(async () => {
  const module = await import('./SettingsDialog');
  await module.prepareSettingsDialog();
  return module;
});

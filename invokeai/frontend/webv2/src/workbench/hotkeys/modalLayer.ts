import { createListenerChannel } from '@workbench/externalStore';

const channel = createListenerChannel();
const activeLayerIds = new Set<string>();

export const registerHotkeyModalLayer = (id: string): (() => void) => {
  activeLayerIds.add(id);
  channel.notify();

  return () => {
    activeLayerIds.delete(id);
    channel.notify();
  };
};

export const isHotkeyModalLayerActive = (): boolean => activeLayerIds.size > 0;

export const subscribeHotkeyModalLayers = channel.subscribe;

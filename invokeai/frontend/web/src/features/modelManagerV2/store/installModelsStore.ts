import { atom } from 'nanostores';

type InstallModelsTabName = 'launchpad' | 'urlOrLocal' | 'huggingface' | 'external' | 'scanFolder' | 'starterModels';

const TAB_TO_INDEX_MAP: Record<InstallModelsTabName, number> = {
  launchpad: 0,
  urlOrLocal: 1,
  huggingface: 2,
  external: 3,
  scanFolder: 4,
  starterModels: 5,
};

export const setInstallModelsTabByName = (tab: InstallModelsTabName) => {
  $installModelsTabIndex.set(TAB_TO_INDEX_MAP[tab]);
};
export const $installModelsTabIndex = atom(0);

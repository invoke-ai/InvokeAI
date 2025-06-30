import { atom } from 'nanostores';

type InstallModelsTabName = 'launchpad' | 'urlOrLocal' | 'huggingface' | 'scanFolder' | 'starterModels';

const TAB_TO_INDEX_MAP: Record<InstallModelsTabName, number> = {
  launchpad: 0,
  urlOrLocal: 1,
  huggingface: 2,
  scanFolder: 3,
  starterModels: 4,
};

export const setInstallModelsTabByName = (tab: InstallModelsTabName) => {
  $installModelsTabIndex.set(TAB_TO_INDEX_MAP[tab]);
};
export const $installModelsTabIndex = atom(0);

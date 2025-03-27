import { atom, computed } from 'nanostores';

export const $isInDeployFlow = atom(false);
export const $outputNodeId = atom<string | null>(null);
export const $isSelectingOutputNode = atom(false);
export const $isReadyToDoValidationRun = computed(
  [$isInDeployFlow, $outputNodeId, $isSelectingOutputNode],
  (isInDeployFlow, outputNodeId, isSelectingOutputNode) => {
    return isInDeployFlow && outputNodeId !== null && !isSelectingOutputNode;
  }
);

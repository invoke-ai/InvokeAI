import { $didStudioInit } from 'app/hooks/useStudioInitAction';
import { atom, computed } from 'nanostores';
import { flushSync } from 'react-dom';

export const $isLayoutLoading = atom(false);
export const setIsLayoutLoading = (isLoading: boolean) => {
  flushSync(() => {
    $isLayoutLoading.set(isLoading);
  });
};
export const $globalIsLoading = computed([$didStudioInit, $isLayoutLoading], (didStudioInit, isLayoutLoading) => {
  return !didStudioInit || isLayoutLoading;
});

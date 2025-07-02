import { deepClone } from 'common/util/deepClone';
import { atom } from 'nanostores';
import type { ImageDTO } from 'services/api/types';

type SuccessState = {
  isSuccess: true;
  isError: false;
  isPending: false;
  result: string;
  error: null;
  imageDTO?: ImageDTO;
};

type ErrorState = {
  isSuccess: false;
  isError: true;
  isPending: false;
  result: null;
  error: Error;
  imageDTO?: ImageDTO;
};

type PendingState = {
  isSuccess: false;
  isError: false;
  isPending: true;
  result: null;
  error: null;
  imageDTO?: ImageDTO;
};

type IdleState = {
  isSuccess: false;
  isError: false;
  isPending: false;
  result: null;
  error: null;
  imageDTO?: ImageDTO;
};

export type PromptExpansionRequestState = IdleState | PendingState | SuccessState | ErrorState;

const IDLE_STATE: IdleState = {
  isSuccess: false,
  isError: false,
  isPending: false,
  result: null,
  error: null,
  imageDTO: undefined,
};

const $state = atom<PromptExpansionRequestState>(deepClone(IDLE_STATE));

const reset = () => {
  $state.set(deepClone(IDLE_STATE));
};

const setPending = (imageDTO?: ImageDTO) => {
  $state.set({
    ...$state.get(),
    isSuccess: false,
    isError: false,
    isPending: true,
    result: null,
    error: null,
    imageDTO,
  });
};

const setSuccess = (result: string) => {
  $state.set({
    ...$state.get(),
    isSuccess: true,
    isError: false,
    isPending: false,
    result,
    error: null,
  });
};

const setError = (error: Error) => {
  $state.set({
    ...$state.get(),
    isSuccess: false,
    isError: true,
    isPending: false,
    result: null,
    error,
  });
};

export const promptExpansionApi = {
  $state,
  reset,
  setPending,
  setSuccess,
  setError,
};

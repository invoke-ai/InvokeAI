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
const buildSuccessState = (result: string): SuccessState => ({
  isSuccess: true,
  isError: false,
  isPending: false,
  result,
  error: null,
});

type ErrorState = {
  isSuccess: false;
  isError: true;
  isPending: false;
  result: null;
  error: Error;
  imageDTO?: ImageDTO;
};
const buildErrorState = (error: Error): ErrorState => ({
  isSuccess: false,
  isError: true,
  isPending: false,
  result: null,
  error,
});

type PendingState = {
  isSuccess: false;
  isError: false;
  isPending: true;
  result: null;
  error: null;
  imageDTO?: ImageDTO;
};
const buildPendingState = (imageDTO?: ImageDTO): PendingState => ({
  isSuccess: false,
  isError: false,
  isPending: true,
  result: null,
  error: null,
  imageDTO,
});

type IdleState = {
  isSuccess: false;
  isError: false;
  isPending: false;
  result: null;
  error: null;
  imageDTO?: ImageDTO;
};
const buildIdleState = (): IdleState => ({
  isSuccess: false,
  isError: false,
  isPending: false,
  result: null,
  error: null,
});

export type PromptExpansionRequestState = IdleState | PendingState | SuccessState | ErrorState;

const $state = atom<PromptExpansionRequestState>(buildIdleState());
const reset = () => {
  $state.set(buildIdleState());
};

const setPending = (imageDTO?: ImageDTO) => {
  const currentState = $state.get() ?? buildPendingState(imageDTO);
  const newState = {
    ...currentState,
    isSuccess: false,
    isError: false,
    isPending: true,
    result: null,
    error: null,
    imageDTO,
  } satisfies PendingState;
  $state.set(newState);
};

const setSuccess = (result: string) => {
  const currentState = $state.get() ?? buildSuccessState(result);
  const newState = {
    ...currentState,
    isSuccess: true,
    isError: false,
    isPending: false,
    result,
    error: null,
  } satisfies SuccessState;
  $state.set(newState);
};

const setError = (error: Error) => {
  const currentState = $state.get() ?? buildErrorState(error);
  const newState = {
    ...currentState,
    isSuccess: false,
    isError: true,
    isPending: false,
    result: null,
    error,
  } satisfies ErrorState;
  $state.set(newState);
};

export const promptExpansionApi = {
  $state,
  reset,
  setPending,
  setSuccess,
  setError,
};

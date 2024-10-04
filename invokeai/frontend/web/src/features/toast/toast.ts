import type { UseToastOptions } from '@invoke-ai/ui-library';
import { createStandaloneToast, theme, TOAST_OPTIONS } from '@invoke-ai/ui-library';
import { nanoid } from 'features/controlLayers/konva/util';
import { map } from 'nanostores';

const toastApi = createStandaloneToast({
  theme: theme,
  defaultOptions: TOAST_OPTIONS.defaultOptions,
}).toast;

// Slightly modified version of UseToastOptions
type ToastConfig = Omit<UseToastOptions, 'id'> & {
  // Only string - Chakra allows numbers
  id?: string;
};

type ToastArg = ToastConfig & {
  /**
   * Whether to append the number of times this toast has been shown to the title. Defaults to true.
   * @example
   * toast({ title: 'Hello', withCount: true });
   * // first toast: 'Hello'
   * // second toast: 'Hello (2)'
   */
  withCount?: boolean;
  /**
   * Whether to update the description when updating the toast. Defaults to true.
   * @example
   * // updateDescription: true
   * toast({ title: 'Hello', description: 'Foo' }); // Foo
   * toast({ title: 'Hello', description: 'Bar' }); // Bar
   * @example
   * // updateDescription: false
   * toast({ title: 'Hello', description: 'Foo' }); // Foo
   * toast({ title: 'Hello', description: 'Bar' }); // Foo
   */
  updateDescription?: boolean;
};

type ToastInternalState = {
  id: string;
  config: ToastConfig;
  count: number;
};

// We expose a limited API for the toast
type ToastApi = {
  getState: () => ToastInternalState | null;
  close: () => void;
  isActive: () => boolean;
};

// Store each toast state by id, allowing toast consumers to not worry about persistent ids and updating and such
const $toastMap = map<Record<string, ToastInternalState | undefined>>({});

// Helpers to get the getters for the toast API
const getIsActive = (id: string) => () => toastApi.isActive(id);
const getClose = (id: string) => () => toastApi.close(id);
const getGetState = (id: string) => () => $toastMap.get()[id] ?? null;

/**
 * Creates a toast with the given config. If the toast with the same id already exists, it will be updated.
 * When a toast is updated, its title, description, status and duration will be overwritten by the new config.
 * Use `updateDescription: false` to keep the description when updating.
 * Set duration to `null` to make the toast persistent.
 * @param arg The toast config.
 * @returns An object with methods to get the toast state, close the toast and check if the toast is active
 */
export const toast = (arg: ToastArg): ToastApi => {
  // All toasts need an id, set a random one if not provided
  const id = arg.id ?? nanoid();
  if (!arg.id) {
    arg.id = id;
  }
  if (arg.withCount === undefined) {
    arg.withCount = true;
  }
  if (arg.updateDescription === undefined) {
    arg.updateDescription = true;
  }
  let state = $toastMap.get()[arg.id];
  if (!state) {
    // First time caller, create and set the state
    state = { id, config: parseConfig(null, id, arg, 1), count: 1 };
    $toastMap.setKey(id, state);
    // Create the toast
    toastApi(state.config);
  } else {
    // This toast is already active, update its state
    state.count += 1;
    state.config = parseConfig(state, id, arg, state.count);
    $toastMap.setKey(id, state);
    // Update the toast itself
    toastApi.update(id, state.config);
  }
  return { getState: getGetState(id), close: getClose(id), isActive: getIsActive(id) };
};

/**
 * Give a toast id, arg and current count, returns the parsed toast config (including dynamic title and description)
 * @param state The current state of the toast or null if it doesn't exist
 * @param id The id of the toast
 * @param arg The arg passed to the toast function
 * @param count The current call count of the toast
 * @returns The parsed toast config
 */
const parseConfig = (state: ToastInternalState | null, id: string, arg: ToastArg, count: number): ToastConfig => {
  const onCloseComplete = () => {
    $toastMap.setKey(id, undefined);
    if (arg.onCloseComplete) {
      arg.onCloseComplete();
    }
  };
  const title = arg.withCount && count > 1 ? `${arg.title} (${count})` : arg.title;
  const description = !arg.updateDescription && state ? state.config.description : arg.description;
  const config: ToastConfig = {
    ...arg,
    title,
    description,
    onCloseComplete,
  };
  return config;
};

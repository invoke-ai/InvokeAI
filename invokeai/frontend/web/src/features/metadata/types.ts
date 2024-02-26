import type { ControlNetConfig, IPAdapterConfig, T2IAdapterConfig } from 'features/controlAdapters/store/types';
import type { O } from 'ts-toolbelt';

/**
 * Renders a value of type T as a React node.
 */
export type MetadataRenderValueFunc<T> = (value: T) => Promise<React.ReactNode>;

/**
 * Gets the label of the current metadata item as a string.
 */
export type MetadataGetLabelFunc = () => string;

export type MetadataParseOptions = {
  toastOnFailure?: boolean;
  toastOnSuccess?: boolean;
};

export type MetadataRecallOptions = MetadataParseOptions;

/**
 * A function that recalls a parsed and validated metadata value.
 *
 * @param value The value to recall.
 * @throws MetadataRecallError if the value cannot be recalled.
 */
export type MetadataRecallFunc<T> = (value: T) => void;

/**
 * An async function that receives metadata and returns a parsed value, throwing if the value is invalid or missing.
 *
 * The function receives an object of unknown type. It is responsible for extracting the relevant data from the metadata
 * and returning a value of type T.
 *
 * The function should throw a MetadataParseError if the metadata is invalid or missing.
 *
 * @param metadata The metadata to parse.
 * @returns A promise that resolves to the parsed value.
 * @throws MetadataParseError if the metadata is invalid or missing.
 */
export type MetadataParseFunc<T = unknown> = (metadata: unknown) => Promise<T>;

/**
 * A function that performs additional validation logic before recalling a metadata value. It is called with a parsed
 * value and should throw if the validation logic fails.
 *
 * This function is used in cases where some additional logic is required before recalling. For example, when recalling
 * a LoRA, we need to check if it is compatible with the current base model.
 *
 * @param value The value to validate.
 * @returns A promise that resolves to the validated value.
 * @throws MetadataRecallError if the value is invalid.
 */
export type MetadataValidateFunc<T> = (value: T) => Promise<T>;

export type MetadataHandlers<TValue = unknown, TItem = unknown> = {
  /**
   * Gets the label of the current metadata item as a string.
   *
   * @returns The label of the current metadata item.
   */
  getLabel: MetadataGetLabelFunc;
  /**
   * An async function that receives metadata and returns a parsed metadata value.
   *
   * @param metadata The metadata to parse.
   * @param withToast Whether to show a toast on success or failure.
   * @returns A promise that resolves to the parsed value.
   * @throws MetadataParseError if the metadata is invalid or missing.
   */
  parse: (metadata: unknown, withToast?: boolean) => Promise<TValue>;
  /**
   * An async function that receives a metadata item and returns a parsed metadata item value.
   *
   * This is only provided if the metadata value is an array.
   *
   * @param item The item to parse. It should be an item from the array.
   * @param withToast Whether to show a toast on success or failure.
   * @returns A promise that resolves to the parsed value.
   * @throws MetadataParseError if the metadata is invalid or missing.
   */
  parseItem?: (item: unknown, withToast?: boolean) => Promise<TItem>;
  /**
   * An async function that recalls a parsed metadata value.
   *
   * This function is only provided if the metadata value can be recalled.
   *
   * @param value The value to recall.
   * @param withToast Whether to show a toast on success or failure.
   * @returns A promise that resolves when the recall operation is complete.
   * @throws MetadataRecallError if the value cannot be recalled.
   */
  recall?: (value: TValue, withToast?: boolean) => Promise<void>;
  /**
   * An async function that recalls a parsed metadata item value.
   *
   * This function is only provided if the metadata value is an array and the items can be recalled.
   *
   * @param item The item to recall. It should be an item from the array.
   * @param withToast Whether to show a toast on success or failure.
   * @returns A promise that resolves when the recall operation is complete.
   * @throws MetadataRecallError if the value cannot be recalled.
   */
  recallItem?: (item: TItem, withToast?: boolean) => Promise<void>;
  /**
   * Renders a parsed metadata value as a React node.
   *
   * @param value The value to render.
   * @returns The rendered value.
   */
  renderValue: MetadataRenderValueFunc<TValue>;
  /**
   * Renders a parsed metadata item value as a React node.
   *
   * @param item The item to render.
   * @returns The rendered item.
   */
  renderItemValue?: MetadataRenderValueFunc<TItem>;
};

// TODO(psyche): The types for item handlers should be able to be inferred from the type of the value:
// type MetadataHandlersInferItem<TValue> = TValue extends Array<infer TItem> ? MetadataParseFunc<TItem> : never
// While this works for the types as expected, I couldn't satisfy TS in the implementations of the handlers.

export type BuildMetadataHandlersArg<TValue, TItem> = {
  parser: MetadataParseFunc<TValue>;
  itemParser?: MetadataParseFunc<TItem>;
  recaller?: MetadataRecallFunc<TValue>;
  itemRecaller?: MetadataRecallFunc<TItem>;
  validator?: MetadataValidateFunc<TValue>;
  itemValidator?: MetadataValidateFunc<TItem>;
  getLabel: MetadataGetLabelFunc;
  renderValue?: MetadataRenderValueFunc<TValue>;
  renderItemValue?: MetadataRenderValueFunc<TItem>;
};

export type BuildMetadataHandlers = <TValue, TItem>(
  arg: BuildMetadataHandlersArg<TValue, TItem>
) => MetadataHandlers<TValue, TItem>;

export type ControlNetConfigMetadata = O.NonNullable<ControlNetConfig, 'model'>;
export type T2IAdapterConfigMetadata = O.NonNullable<T2IAdapterConfig, 'model'>;
export type IPAdapterConfigMetadata = O.NonNullable<IPAdapterConfig, 'model'>;
export type AnyControlAdapterConfigMetadata =
  | ControlNetConfigMetadata
  | T2IAdapterConfigMetadata
  | IPAdapterConfigMetadata;

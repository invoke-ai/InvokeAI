import React, { PropsWithChildren } from 'react';
import { IAIPopoverProps } from '../web/src/common/components/IAIPopover';
import { IAIIconButtonProps } from '../web/src/common/components/IAIIconButton';

export {};

declare module 'redux-socket.io-middleware';

declare global {
  /* eslint-disable @typescript-eslint/no-explicit-any */
  interface Array<T> {
    /**
     * Returns the value of the last element in the array where predicate is true, and undefined
     * otherwise.
     * @param predicate findLast calls predicate once for each element of the array, in descending
     * order, until it finds one where predicate returns true. If such an element is found, findLast
     * immediately returns that element value. Otherwise, findLast returns undefined.
     * @param thisArg If provided, it will be used as the this value for each invocation of
     * predicate. If it is not provided, undefined is used instead.
     */
    findLast<S extends T>(
      predicate: (value: T, index: number, array: T[]) => value is S,
      thisArg?: any
    ): S | undefined;
    findLast(
      predicate: (value: T, index: number, array: T[]) => unknown,
      thisArg?: any
    ): T | undefined;

    /**
     * Returns the index of the last element in the array where predicate is true, and -1
     * otherwise.
     * @param predicate findLastIndex calls predicate once for each element of the array, in descending
     * order, until it finds one where predicate returns true. If such an element is found,
     * findLastIndex immediately returns that element index. Otherwise, findLastIndex returns -1.
     * @param thisArg If provided, it will be used as the this value for each invocation of
     * predicate. If it is not provided, undefined is used instead.
     */
    findLastIndex(
      predicate: (value: T, index: number, array: T[]) => unknown,
      thisArg?: any
    ): number;
  }
  /* eslint-enable @typescript-eslint/no-explicit-any */
}

declare module '@invoke-ai/invoke-ai-ui' {
  declare class ThemeChanger extends React.Component<ThemeChangerProps> {
    public constructor(props: ThemeChangerProps);
  }

  declare class InvokeAiLogoComponent extends React.Component<InvokeAILogoComponentProps> {
    public constructor(props: InvokeAILogoComponentProps);
  }

  declare class IAIPopover extends React.Component<IAIPopoverProps> {
    public constructor(props: IAIPopoverProps);
  }

  declare class IAIIconButton extends React.Component<IAIIconButtonProps> {
    public constructor(props: IAIIconButtonProps);
  }

  declare class SettingsModal extends React.Component<SettingsModalProps> {
    public constructor(props: SettingsModalProps);
  }
}

declare function Invoke(props: PropsWithChildren): JSX.Element;

export {
  ThemeChanger,
  InvokeAiLogoComponent,
  IAIPopover,
  IAIIconButton,
  SettingsModal,
};
export = Invoke;

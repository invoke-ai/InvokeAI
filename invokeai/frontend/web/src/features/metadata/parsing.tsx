import type { AppStore } from 'app/store/store';
import { type Atom, atom } from 'nanostores';
import type { ReactNode } from 'react';

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
type Item<T> = T extends any[] ? T[number] : T;

type UnparsedData = {
  isParsed: false;
  isSuccess: false;
  isError: false;
  raw: unknown;
  parsed: null;
  error: null;
};

type ParsedSuccessData<T> = {
  isParsed: true;
  isSuccess: true;
  isError: false;
  raw: unknown;
  parsed: T;
  error: null;
};

type ParsedErrorData = {
  isParsed: true;
  isSuccess: false;
  isError: true;
  raw: unknown;
  parsed: null;
  error: Error;
};

type Data<T> = UnparsedData | ParsedSuccessData<T> | ParsedErrorData;

abstract class MetadataParser<T> {
  $data: Atom<Data<T>>;
  store: AppStore;

  abstract extract: (metadata: unknown) => Promise<unknown>;
  abstract parse: (metadata: unknown) => Promise<T>;
  abstract recall: (data: Item<T>) => Promise<void>;
  abstract renderLabel: (data: Item<T>) => ReactNode;
  abstract renderValue: (data: Item<T>) => ReactNode;

  constructor(store: AppStore) {
    this.$data = atom(MetadataParser.getInitialData());
    this.store = store;
  }

  static getInitialData = (): UnparsedData => {
    return {
      isParsed: false,
      isSuccess: false,
      isError: false,
      raw: null,
      parsed: null,
      error: null,
    };
  };
}

export class PositivePromptParser extends MetadataParser<string> {
  constructor(store: AppStore) {
    super(store);
  }

  extract = (metadata: unknown) => {
    return Promise.resolve(metadata);
  };

  parse = (_metadata: unknown) => {
    return Promise.resolve('test');
  };

  recall = (_data: string) => {
    return Promise.resolve();
  };

  renderLabel = (data: string) => {
    return <div>{data}</div>;
  };

  renderValue = (data: string) => {
    return <div>{data}</div>;
  };
}

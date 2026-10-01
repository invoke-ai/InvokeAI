export type TokenizerFamily = 'clip' | 't5' | 'qwen' | 'estimate';

export type TokenCountResult = {
  count: number;
  limit: number;
  tokenizerFamily: TokenizerFamily;
  isNearLimit: boolean;
  isOverLimit: boolean;
};

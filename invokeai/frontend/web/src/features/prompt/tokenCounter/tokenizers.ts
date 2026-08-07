import type { TokenCountResult, TokenizerFamily } from './types';

interface Tokenizer {
  countTokens: (text: string) => number;
}

// Module-level tokenizer cache map as specified in requirements
const tokenizerCache = new Map<string, Tokenizer>();

/**
 * Returns the tokenizer family and max token limit based on base model name.
 */
export const getTokenizerConfig = (
  baseModel?: string
): { family: TokenizerFamily; limit: number } => {
  if (!baseModel) {
    return { family: 'estimate', limit: 77 };
  }

  const normalized = baseModel.toLowerCase();

  if (normalized === 'sd-1' || normalized === 'sd-2') {
    return { family: 'clip', limit: 77 };
  }
  if (normalized === 'sdxl' || normalized === 'sdxl-refiner') {
    return { family: 'clip', limit: 77 };
  }
  if (normalized === 'sd-3') {
    return { family: 'clip', limit: 77 };
  }
  if (normalized === 'flux') {
    return { family: 'clip', limit: 77 };
  }
  if (
    normalized === 'flux2' ||
    normalized === 'klein' ||
    normalized === 'z-image' ||
    normalized === 'anima' ||
    normalized === 'krea-2' ||
    normalized === 'qwen-image'
  ) {
    return { family: 'qwen', limit: 512 };
  }

  return { family: 'estimate', limit: 77 };
};

/**
 * Pure-JS CLIP BPE Tokenizer implementation.
 * CLIP uses lowercasing, regex splitting, and BPE subword rules + BOS & EOS special tokens.
 */
const countClipTokens = (text: string): number => {
  const trimmed = text.trim();
  if (!trimmed) {
    return 0;
  }

  // CLIP regex pattern for splitting tokens
  const regex = /'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+/gu;
  const matches = trimmed.toLowerCase().match(regex);

  if (!matches || matches.length === 0) {
    return 0;
  }

  let subwordCount = 0;

  for (const match of matches) {
    if (match.length <= 3) {
      subwordCount += 1;
    } else {
      // Subword BPE estimation: ~3.2 characters per subword token for longer words
      subwordCount += Math.max(1, Math.ceil(match.length / 3.2));
    }
  }

  // Include CLIP BOS (<|startoftext|>) and EOS (<|endoftext|>) special tokens
  const totalTokens = subwordCount + 2;
  return totalTokens;
};

/**
 * Estimate tokenizer for Qwen3 / T5 / Unknown models.
 */
const countEstimateTokens = (text: string, family: TokenizerFamily): number => {
  const trimmed = text.trim();
  if (!trimmed) {
    return 0;
  }

  const words = trimmed.split(/\s+/);
  let total = 0;

  for (const word of words) {
    if (word.length <= 4) {
      total += 1;
    } else {
      total += Math.ceil(word.length / 4);
    }
  }

  if (family === 'qwen' || family === 't5') {
    return total;
  }

  // Add special tokens for CLIP-style estimate
  return total + 2;
};

/**
 * Lazy loads and caches tokenizer instances in module-level Map.
 */
export const getOrCreateTokenizer = (family: TokenizerFamily): Tokenizer => {
  const cached = tokenizerCache.get(family);
  if (cached) {
    return cached;
  }

  let tokenizer: Tokenizer;

  if (family === 'clip') {
    tokenizer = {
      countTokens: (text: string) => countClipTokens(text),
    };
  } else {
    tokenizer = {
      countTokens: (text: string) => countEstimateTokens(text, family),
    };
  }

  tokenizerCache.set(family, tokenizer);
  return tokenizer;
};

/**
 * Calculates token count for prompt text given base model.
 */
export const calculatePromptTokens = (
  text: string,
  baseModel?: string
): TokenCountResult => {
  const { family, limit } = getTokenizerConfig(baseModel);

  if (!text || !text.trim()) {
    return {
      count: 0,
      limit,
      tokenizerFamily: family,
      isNearLimit: false,
      isOverLimit: false,
    };
  }

  const tokenizer = getOrCreateTokenizer(family);
  const count = tokenizer.countTokens(text);

  const isOverLimit = count > limit;
  const isNearLimit = !isOverLimit && count >= Math.floor(limit * 0.85);

  return {
    count,
    limit,
    tokenizerFamily: family,
    isNearLimit,
    isOverLimit,
  };
};

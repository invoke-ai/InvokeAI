import { describe, expect, it } from 'vitest';

import { formatBytes, getModelFormatLabel, getModelTypeLabel } from './taxonomy';

describe('formatBytes', () => {
  it('renders a dash for missing or invalid sizes', () => {
    expect(formatBytes(null)).toBe('—');
    expect(formatBytes(undefined)).toBe('—');
    expect(formatBytes(-1)).toBe('—');
    expect(formatBytes(Number.NaN)).toBe('—');
  });

  it('scales across unit boundaries', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(1023)).toBe('1023 B');
    expect(formatBytes(1024)).toBe('1.0 KB');
    expect(formatBytes(1024 * 1024)).toBe('1.0 MB');
    expect(formatBytes(1.5 * 1024 * 1024 * 1024)).toBe('1.5 GB');
  });
});

describe('label fallbacks', () => {
  it('title-cases unknown open-union values', () => {
    expect(getModelTypeLabel('some_new_type')).toBe('Some New Type');
    expect(getModelFormatLabel('some_new-format')).toBe('Some New Format');
  });

  it('prefers curated labels for known values', () => {
    expect(getModelTypeLabel('lora')).toBe('LoRA');
    expect(getModelTypeLabel('qwen3_vl_encoder')).toBe('Qwen3 VL Encoder');
    expect(getModelTypeLabel('gemma2_encoder')).toBe('Gemma 2 Encoder');
    expect(getModelTypeLabel('pid_decoder')).toBe('PiD Decoder');
    expect(getModelFormatLabel('gguf_quantized')).toBe('GGUF');
  });
});

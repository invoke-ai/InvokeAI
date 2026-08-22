import { describe, expect, it } from 'vitest';

import {
  EDITABLE_MODEL_FORMATS,
  getModelSourceHref,
  formatBytes,
  getModelFormatLabel,
  getModelTypeLabel,
  getModelVariantLabel,
  getVariantOptionsFor,
} from './taxonomy';

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
    expect(getModelFormatLabel('sdnq_quantized')).toBe('SDNQ');
  });
});

describe('variant options', () => {
  it('offers main-model variants keyed by base', () => {
    expect(getVariantOptionsFor('sd-1', 'main')).toEqual(['normal', 'inpaint']);
    expect(getVariantOptionsFor('sd-2', 'main')).toEqual(['normal', 'inpaint', 'depth']);
    expect(getVariantOptionsFor('flux', 'main')).toEqual(['schnell', 'dev', 'dev_fill']);
    expect(getVariantOptionsFor('wan', 'main')).toEqual(['t2v_a14b', 'i2v_a14b', 'ti2v_5b']);
  });

  it('distinguishes wan main and wan lora variants', () => {
    expect(getVariantOptionsFor('wan', 'lora')).toEqual(['a14b', '5b']);
    expect(getVariantOptionsFor('sdxl', 'lora')).toEqual([]);
  });

  it('offers per-type variants for encoder-style types regardless of base', () => {
    expect(getVariantOptionsFor('any', 'clip_embed')).toEqual(['large', 'gigantic']);
    expect(getVariantOptionsFor('flux2', 'qwen3_encoder')).toEqual(['qwen3_4b', 'qwen3_8b', 'qwen3_06b']);
  });

  it('returns empty for pairs with no variant concept, enabling free text', () => {
    expect(getVariantOptionsFor('sdxl', 'vae')).toEqual([]);
    expect(getVariantOptionsFor('unknown', 'main')).toEqual([]);
  });

  it('labels known variants and title-cases unknown ones', () => {
    expect(getModelVariantLabel('dev_fill')).toBe('FLUX Dev - Fill');
    expect(getModelVariantLabel('some_new_variant')).toBe('Some New Variant');
  });

  it('keeps unknown and external_api out of the assignable formats', () => {
    expect(EDITABLE_MODEL_FORMATS).not.toContain('unknown');
    expect(EDITABLE_MODEL_FORMATS).not.toContain('external_api');
    expect(EDITABLE_MODEL_FORMATS).toContain('checkpoint');
    expect(EDITABLE_MODEL_FORMATS).toContain('diffusers');
  });
});

describe('getModelSourceHref', () => {
  it('links http(s) sources directly and repo ids to their HuggingFace page', () => {
    expect(getModelSourceHref('https://civitai.com/api/download/models/1', 'url')).toBe(
      'https://civitai.com/api/download/models/1'
    );
    expect(getModelSourceHref('owner/repo', 'hf_repo_id')).toBe('https://huggingface.co/owner/repo');
    expect(getModelSourceHref('owner/repo:fp16:path/x.safetensors', 'hf_repo_id')).toBe(
      'https://huggingface.co/owner/repo'
    );
  });

  it('returns null for local paths and unlinkable sources', () => {
    expect(getModelSourceHref('/models/x.safetensors', 'path')).toBeNull();
    expect(getModelSourceHref('C:\\models\\x.safetensors', 'path')).toBeNull();
    expect(getModelSourceHref('some-name', 'unknown')).toBeNull();
  });
});

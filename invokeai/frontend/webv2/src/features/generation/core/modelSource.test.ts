import { describe, expect, it } from 'vitest';

import { isModelFromRepo } from './modelSource';

const REPO = 'google/gemma-4-E4B-it';

describe('isModelFromRepo', () => {
  it('matches the repo in its Hugging Face forms', () => {
    expect(isModelFromRepo('google/gemma-4-E4B-it', REPO)).toBe(true);
    expect(isModelFromRepo('Google/Gemma-4-e4b-IT', REPO)).toBe(true);
    expect(isModelFromRepo('google/gemma-4-E4B-it::text_encoder', REPO)).toBe(true);
    expect(isModelFromRepo('https://huggingface.co/google/gemma-4-E4B-it/tree/main', REPO)).toBe(true);
  });

  it('matches a local folder named like the repo', () => {
    expect(isModelFromRepo('E:/ModelStuffSSD/gemma-4-E4B-it', REPO)).toBe(true);
    expect(isModelFromRepo('E:\\ModelStuffSSD\\GEMMA-4-E4B-IT\\', REPO)).toBe(true);
    expect(isModelFromRepo('/models/gemma-4-E4B-it', REPO)).toBe(true);
  });

  it('refuses other repos, other folders and a missing source', () => {
    expect(isModelFromRepo('google/gemma-4-E2B-it', REPO)).toBe(false);
    expect(isModelFromRepo('someone/gemma-4-E4B-it', REPO)).toBe(false);
    expect(isModelFromRepo('E:/ModelStuffSSD/gemma-4-E4B-it-q4', REPO)).toBe(false);
    expect(isModelFromRepo('E:/ModelStuffSSD/gemma-4-E4B-it/model.safetensors', REPO)).toBe(false);
    expect(isModelFromRepo(null, REPO)).toBe(false);
  });
});

import { describe, expect, it } from 'vitest';

import {
  applyPromptTemplate,
  getEffectivePrompts,
  getPromptTemplateChunks,
  type PromptTemplateSnapshot,
  sanitizePromptTemplateSnapshot,
  syncPromptTemplateWithCatalog,
} from './promptTemplates';

const template = (overrides: Partial<PromptTemplateSnapshot> = {}): PromptTemplateSnapshot => ({
  id: 'template-1',
  name: 'Cinematic',
  negativePrompt: '',
  positivePrompt: '',
  ...overrides,
});

describe('applyPromptTemplate', () => {
  it('substitutes the authored prompt at the placeholder', () => {
    expect(applyPromptTemplate('{prompt}. photography, bokeh', 'a cat')).toBe('a cat. photography, bokeh');
  });

  it('appends after a single space when there is no placeholder', () => {
    expect(applyPromptTemplate('oil painting', 'a cat')).toBe('a cat oil painting');
  });

  // The legacy `String.replace` takes a string pattern, so only the first
  // placeholder is an insertion point. `getPromptTemplateChunks` has to agree.
  it('substitutes only the first of several placeholders', () => {
    expect(applyPromptTemplate('{prompt} in the style of {prompt}', 'a cat')).toBe('a cat in the style of {prompt}');
  });

  it('keeps the placeholder at either end of the template', () => {
    expect(applyPromptTemplate('{prompt}, 8k', 'a cat')).toBe('a cat, 8k');
    expect(applyPromptTemplate('a painting of {prompt}', 'a cat')).toBe('a painting of a cat');
  });

  // Whitespace is literal on both sides, including the leading space an empty
  // prompt produces. The legacy client emits exactly this, so a template written
  // there must render identically here.
  it('is whitespace-literal with an empty authored prompt', () => {
    expect(applyPromptTemplate('oil painting', '')).toBe(' oil painting');
    expect(applyPromptTemplate('{prompt}, 8k', '')).toBe(', 8k');
  });

  it('returns the authored prompt when the template is empty', () => {
    expect(applyPromptTemplate('', 'a cat')).toBe('a cat ');
  });
});

describe('getEffectivePrompts', () => {
  it('passes the authored prompts through when no template is active', () => {
    expect(getEffectivePrompts({ negativePrompt: 'blurry', positivePrompt: 'a cat', promptTemplate: null })).toEqual({
      negativePrompt: 'blurry',
      positivePrompt: 'a cat',
    });
  });

  it('merges the positive and negative sides independently', () => {
    expect(
      getEffectivePrompts({
        negativePrompt: 'blurry',
        positivePrompt: 'a cat',
        promptTemplate: template({ negativePrompt: '{prompt}, lowres', positivePrompt: '{prompt}, cinematic' }),
      })
    ).toEqual({ negativePrompt: 'blurry, lowres', positivePrompt: 'a cat, cinematic' });
  });

  // A template side left blank still appends its (empty) text, exactly as legacy
  // does — the trailing space is the observable consequence and is pinned here so
  // a "tidy up the whitespace" refactor has to be deliberate.
  it('appends an empty template side rather than skipping it', () => {
    expect(
      getEffectivePrompts({
        negativePrompt: 'blurry',
        positivePrompt: 'a cat',
        promptTemplate: template({ positivePrompt: 'cinematic' }),
      })
    ).toEqual({ negativePrompt: 'blurry ', positivePrompt: 'a cat cinematic' });
  });

  // The merged text is what the expansion gate sees, so a template may make an
  // otherwise-literal prompt dynamic. Guarding this here keeps the ordering
  // requirement visible next to the function that depends on it.
  it('can introduce dynamic syntax the authored prompt did not have', () => {
    expect(
      getEffectivePrompts({
        negativePrompt: '',
        positivePrompt: 'a cat',
        promptTemplate: template({ positivePrompt: '{prompt}, {red|green} tint' }),
      }).positivePrompt
    ).toBe('a cat, {red|green} tint');
  });
});

describe('getPromptTemplateChunks', () => {
  // Not `['', 'a cat', '']`: an empty template side still appends, so the middle
  // chunk carries the trailing space the merge produces.
  it('keeps the appended space when the template side is empty', () => {
    expect(getPromptTemplateChunks('a cat', '')).toEqual(['', 'a cat ', '']);
  });

  it('trails the template after the prompt when there is no placeholder', () => {
    expect(getPromptTemplateChunks('a cat', 'oil painting')).toEqual(['', 'a cat ', 'oil painting']);
  });

  it('splits around the placeholder', () => {
    expect(getPromptTemplateChunks('a cat', 'a photo of {prompt}, 8k')).toEqual(['a photo of ', 'a cat', ', 8k']);
  });

  it('rejoins later placeholders into the trailing chunk', () => {
    expect(getPromptTemplateChunks('a cat', '{prompt} by {prompt}')).toEqual(['', 'a cat', ' by {prompt}']);
  });

  // The chunks are only ever rendered, never submitted, so their concatenation
  // has to equal what `applyPromptTemplate` produces or the preview lies.
  //
  // The prompts here are the interesting half: the chunks split (literal) while
  // the merge replaces, and `$&` and friends are special in a replacement string
  // whatever the pattern is. Holding this over one prompt let that diverge.
  it('concatenates to the merged prompt', () => {
    const prompts = ['a cat', 'a poster reading $$$', 'x $& y', "x $' y", 'x $` y', 'a $1 note'];

    for (const templatePrompt of ['', 'oil painting', 'a photo of {prompt}, 8k', '{prompt} by {prompt}']) {
      for (const prompt of prompts) {
        expect(getPromptTemplateChunks(prompt, templatePrompt).join('')).toBe(
          applyPromptTemplate(templatePrompt, prompt)
        );
      }
    }
  });
});

describe('applyPromptTemplate', () => {
  // Regression: the prompt was passed as a replacement *string*, so `$&` put the
  // placeholder back and the expander turned it into the word "prompt", while
  // `$$` swallowed a character outright.
  it('substitutes the prompt literally, dollars and all', () => {
    expect(applyPromptTemplate('a photo of {prompt}, 8k', 'x $& y')).toBe('a photo of x $& y, 8k');
    expect(applyPromptTemplate('a photo of {prompt}, 8k', 'a poster reading $$$')).toBe(
      'a photo of a poster reading $$$, 8k'
    );
    expect(applyPromptTemplate('a photo of {prompt}, 8k', "x $' y")).toBe("a photo of x $' y, 8k");
  });

  it('substitutes only the first placeholder', () => {
    expect(applyPromptTemplate('{prompt} by {prompt}', 'a cat')).toBe('a cat by {prompt}');
  });

  it('appends a template that carries no placeholder', () => {
    expect(applyPromptTemplate('oil painting', 'a cat')).toBe('a cat oil painting');
  });
});

describe('sanitizePromptTemplateSnapshot', () => {
  it('reads a well-formed snapshot', () => {
    const snapshot = template({ negativePrompt: 'blurry', positivePrompt: '{prompt}, 8k' });

    expect(sanitizePromptTemplateSnapshot({ ...snapshot })).toEqual(snapshot);
  });

  it('defaults missing prompt sides to empty strings', () => {
    expect(sanitizePromptTemplateSnapshot({ id: 'a', name: 'A' })).toEqual({
      id: 'a',
      name: 'A',
      negativePrompt: '',
      positivePrompt: '',
    });
  });

  it('rejects values with no usable identity', () => {
    expect(sanitizePromptTemplateSnapshot(null)).toBeNull();
    expect(sanitizePromptTemplateSnapshot('template-1')).toBeNull();
    expect(sanitizePromptTemplateSnapshot([])).toBeNull();
    expect(sanitizePromptTemplateSnapshot({ id: '', name: 'A' })).toBeNull();
    expect(sanitizePromptTemplateSnapshot({ name: 'A' })).toBeNull();
    expect(sanitizePromptTemplateSnapshot({ id: 'a' })).toBeNull();
  });
});

describe('syncPromptTemplateWithCatalog', () => {
  it('picks up an upstream edit', () => {
    const stored = template({ positivePrompt: '{prompt}, cinematic' });
    const edited = template({ name: 'Cinematic v2', positivePrompt: '{prompt}, cinematic, 35mm' });

    expect(syncPromptTemplateWithCatalog(stored, [edited])).toEqual(edited);
  });

  // The catalog holds records, which carry `isDefault` and a host-specific
  // `imageUrl`. Adopting one wholesale persisted both into project state and
  // into every queue item, and put five keys on something the canonical check
  // requires to have four.
  it('takes only the snapshot fields from the catalog entry', () => {
    const stored = template({ positivePrompt: '{prompt}, cinematic' });
    const record = {
      ...template({ positivePrompt: '{prompt}, cinematic, 35mm' }),
      imageUrl: 'http://some-host:9090/api/v1/style_presets/t1/image',
      isDefault: false,
    };

    expect(syncPromptTemplateWithCatalog(stored, [record])).toEqual({
      ...stored,
      positivePrompt: '{prompt}, cinematic, 35mm',
    });
  });

  // The callers diff with `Object.is`, so an equal-but-new object each pass
  // would commit forever.
  it('returns the stored object by reference when nothing changed', () => {
    const stored = template({ positivePrompt: '{prompt}, cinematic' });

    expect(syncPromptTemplateWithCatalog(stored, [{ ...stored }])).toBe(stored);
  });

  // A catalog that is empty because it is still loading, or because another
  // user's template is not visible, must not silently change what generates.
  it('keeps a template the catalog does not carry', () => {
    const stored = template();

    expect(syncPromptTemplateWithCatalog(stored, [])).toBe(stored);
    expect(syncPromptTemplateWithCatalog(stored, [template({ id: 'other' })])).toBe(stored);
  });

  it('stays null when nothing is active', () => {
    expect(syncPromptTemplateWithCatalog(null, [template()])).toBeNull();
  });
});
